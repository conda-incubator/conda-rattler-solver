from __future__ import annotations

import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from string import hexdigits
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import rattler
from conda.base.constants import REPODATA_FN
from conda.base.context import context
from conda.common.io import DummyExecutor, ThreadLimitedThreadPoolExecutor
from conda.common.url import path_to_url, remove_auth, split_anaconda_token
from conda.core.package_cache_data import PackageCacheData
from conda.core.subdir_data import SubdirData
from conda.gateways.repodata.shards import build_repodata_subset
from conda.models.channel import Channel

try:
    from conda.common.serialize.json import dumps as json_dump
except ImportError:
    from conda.common.serialize import json_dump

from .utils import empty_repodata_dict, rattler_record_to_conda_record

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from conda.common.path import PathsType
    from conda.gateways.repodata.shards import ShardBase
    from conda.models.match_spec import MatchSpec
    from conda.models.records import PackageCacheRecord, PackageRecord

    from .state import SolverInputState

log = logging.getLogger(f"conda.{__name__}")


def _is_sharded_repodata_enabled() -> bool:
    return context.plugins.use_sharded_repodata is True  # type: ignore[attr-defined]


def _rattler_parse_dep_name(dep: str) -> str:
    return str(rattler.MatchSpec(dep).name)


@dataclass
class _ChannelRepoInfo:
    "A dataclass mapping conda Channels, rattler.SparseRepoData, URLs and JSON paths"

    channel: Channel | None
    repo: rattler.SparseRepoData
    full_url: str
    noauth_url: str
    local_json: str | None


class RattlerIndexHelper:
    def __init__(
        self,
        channels: Iterable[Channel | str] = None,
        subdirs: Iterable[str] = None,
        repodata_fn: str = REPODATA_FN,
        pkgs_dirs: PathsType = (),
        in_state: SolverInputState | None = None,
    ):
        self._unlink_on_del: list[Path] = []

        raw_channels = context.channels if channels is None else channels
        platform_less_channels: list[Channel] = []
        for c in raw_channels:
            ch = Channel(c)
            if ch.platform:
                log.info(
                    "Platform-aware channels are not supported. "
                    "Ignoring platform %s from channel %s. "
                    "Use subdirs keyword if necessary.",
                    ch.platform,
                    ch,
                )
                ch = Channel(**{k: v for k, v in ch.dump().items() if k != "platform"})
            platform_less_channels.append(ch)
        self._channels = platform_less_channels
        self._subdirs = context.subdirs if subdirs is None else subdirs
        self._repodata_fn = repodata_fn
        self.in_state = in_state

        self._index: dict[str, _ChannelRepoInfo] = {}
        self._index.update(self._load_channels())
        if pkgs_dirs:
            self._index.update(
                {info.noauth_url: info for info in self._load_pkgs_cache(pkgs_dirs)}
            )

    @classmethod
    def from_platform_aware_channel(cls, channel: Channel) -> Self:
        if not channel.platform:
            raise ValueError(f"Channel {channel} must define 'platform' attribute.")
        subdir = channel.platform
        channel = Channel(**{k: v for k, v in channel.dump().items() if k != "platform"})
        return cls(channels=(channel,), subdirs=(subdir,))

    @property
    def channels(self) -> list[Channel]:
        return [Channel(c) for c in self._channels]

    def reload_channel(self, channel: Channel) -> None:
        urls = {}
        for url in channel.urls(with_credentials=False, subdirs=self._subdirs):
            for repo_info in self._index.values():
                if repo_info.noauth_url == url:
                    log.debug("Reloading repo %s", repo_info.noauth_url)
                    urls[repo_info.full_url] = channel
                    break
        for new_repo_info in self._load_channels(urls).values():
            for repo_info in self._index.values():
                if repo_info.noauth_url == new_repo_info.noauth_url:
                    repo_info.repo.close()
                    repo_info.repo = new_repo_info.repo
                    break

    def n_packages(
        self,
        repos: Iterable[_ChannelRepoInfo] | None = None,
        filter_: callable | None = None,
    ) -> int:
        count = 0
        if filter_ is not None:
            for info in repos or self._index.values():
                for record in info.repo.load_all_records(self._package_format):
                    if filter_(record):
                        count += 1
        else:
            for info in repos or self._index.values():
                count += info.repo.record_count(self._package_format)
        return count

    def get_info(self, key: str) -> _ChannelRepoInfo:
        if not key.startswith("file://"):
            # The conda functions (specifically remove_auth) assume the input
            # is a url; a file uri on windows with a drive letter messes them up.
            # For the rest, we remove all forms of authentication
            key = split_anaconda_token(remove_auth(key))[0]
        return self._index[key]

    def _fetch_channel(self, url: str) -> tuple[str, os.PathLike]:
        channel = Channel.from_url(url)
        if not channel.subdir:
            raise ValueError(f"Channel URLs must specify a subdir! Provided: {url}")

        if "PYTEST_CURRENT_TEST" in os.environ:
            # Workaround some testing issues - TODO: REMOVE
            # Fix conda.testing.helpers._patch_for_local_exports by removing last line
            maybe_cached = SubdirData._cache_.get((url, self._repodata_fn))
            if maybe_cached and maybe_cached._mtime == float("inf"):
                del SubdirData._cache_[(url, self._repodata_fn)]
            # /Workaround

        log.debug("Fetching %s with SubdirData.repo_fetch", channel)
        subdir_data = SubdirData(channel, repodata_fn=self._repodata_fn)
        json_path, _ = subdir_data.repo_fetch.fetch_latest_path()

        return url, json_path

    def _json_path_to_repo_info(self, url: str, json_path: str) -> _ChannelRepoInfo:
        channel = Channel.from_url(url)
        noauth_url = channel.urls(with_credentials=False, subdirs=(channel.subdir,))[0]
        noauth_url_sans_subdir = noauth_url.rsplit("/", 1)[0]
        json_path = Path(json_path)
        if (
            sys.platform == "win32"
            and os.environ.get("CI")
            and os.environ.get("PYTEST_CURRENT_TEST")
        ):
            # TODO: Investigate why we need this race condition workaround on Windows CI only
            random_hex = "".join(random.choices(hexdigits, k=6)).lower()
            path_copy = json_path.parent / f"{json_path.stem}.copy-{random_hex}.json"
            shutil.copy(json_path, path_copy)
            json_path = path_copy
            self._unlink_on_del.append(path_copy)
        # TODO: Support multichannel https://github.com/conda/rattler/issues/1327
        rattler_channel = rattler.Channel(noauth_url_sans_subdir)
        repo = rattler.SparseRepoData(rattler_channel, channel.subdir, json_path)
        return _ChannelRepoInfo(
            repo=repo,
            channel=channel,
            full_url=url,
            noauth_url=noauth_url,
            local_json=json_path,
        )

    def _urls_from_channels(self, channels: Iterable[Channel | str] | None = None) -> tuple[str]:
        # 1. Obtain and deduplicate URLs from channels
        urls = []
        seen_noauth = set()
        for _c in channels or self._channels:
            c = Channel(_c)
            noauth_urls = c.urls(with_credentials=False, subdirs=self._subdirs)
            if seen_noauth.issuperset(noauth_urls):
                continue
            if c.auth or c.token:  # authed channel always takes precedence
                urls += Channel(c).urls(with_credentials=True, subdirs=self._subdirs)
                seen_noauth.update(noauth_urls)
                continue
            # at this point, we are handling an unauthed channel; in some edge cases,
            # an auth'd variant of the same channel might already be present in `urls`.
            # we only add them if we haven't seen them yet
            for url in noauth_urls:
                if url not in seen_noauth:
                    urls.append(url)
                    seen_noauth.add(url)

        return tuple(dict.fromkeys(urls))  # de-duplicate

    @staticmethod
    def _channel_urls(subdirs: Iterable[str], channels: list[Channel]) -> dict[str, Channel]:
        urls: dict[str, Channel] = {}
        seen_noauth: set[str] = set()
        channels_with_subdirs: list[Channel] = []
        for channel in channels:
            for url in channel.urls(with_credentials=True, subdirs=subdirs):
                channels_with_subdirs.append(Channel(url))
        for channel in channels_with_subdirs:
            noauth_urls = [
                url for url in channel.urls(with_credentials=False) if url.endswith(channel.subdir)
            ]
            if seen_noauth.issuperset(noauth_urls):
                continue
            auth_urls = [
                url.replace(" ", "%20")
                for url in channel.urls(with_credentials=True)
                if url.endswith(tuple(subdirs))
            ]
            if noauth_urls != auth_urls:
                urls.update({url: channel for url in auth_urls})
                seen_noauth.update(noauth_urls)
                continue
            for url in noauth_urls:
                if url not in seen_noauth:
                    urls[url] = channel
                    seen_noauth.add(url)
        return urls

    @staticmethod
    def _encoded_urls_to_channels(urls_to_channel: dict[str, Channel]) -> dict[str, Channel]:
        encoded: dict[str, Channel] = {}
        for url, channel in urls_to_channel.items():
            if url.startswith("file://"):
                url = url.replace(" ", "%20")
            encoded[url] = channel
        return encoded

    def _load_index_from_shard_subset(
        self, channel_data: dict[str, ShardBase]
    ) -> dict[str, _ChannelRepoInfo]:
        index: dict[str, _ChannelRepoInfo] = {}
        for channel_url, shardlike in channel_data.items():
            repodata = shardlike.build_repodata()
            with NamedTemporaryFile(
                suffix=".json", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(json_dump(repodata))
                path = tmp.name
            self._unlink_on_del.append(Path(path))
            info = self._json_path_to_repo_info(channel_url, path)
            index[info.noauth_url] = info
        return index

    def _load_channels(
        self, urls: dict[str, Channel] | Iterable[str] | None = None
    ) -> dict[str, _ChannelRepoInfo]:
        urls_to_channel: dict[str, Channel] | None = None
        if urls is None:
            urls_to_channel = self._encoded_urls_to_channels(
                self._channel_urls(self._subdirs, self.channels)
            )
            url_tuple = tuple(urls_to_channel.keys())
        elif isinstance(urls, dict):
            urls_to_channel = self._encoded_urls_to_channels(urls)
            url_tuple = tuple(urls.keys())
        else:
            url_tuple = tuple(urls)

        if self.in_state and _is_sharded_repodata_enabled() and urls_to_channel is not None:
            root_packages = (*self.in_state.installed.keys(), *self.in_state.requested)
            channel_data = build_repodata_subset(
                root_packages,
                urls_to_channel,
                parse_dep_name=_rattler_parse_dep_name,
            )
            if channel_data is not None:
                log.debug("Loaded %d channel(s) from sharded repodata", len(channel_data))
                return self._load_index_from_shard_subset(channel_data)
            log.debug("No sharded channels available; falling back to classic repodata fetch.")

        Executor = (
            DummyExecutor
            if context.debug or context.repodata_threads == 1
            else partial(ThreadLimitedThreadPoolExecutor, max_workers=context.repodata_threads)
        )
        with Executor() as executor:
            pairs = executor.map(self._fetch_channel, url_tuple)
            jsons = {url: str(path) for (url, path) in pairs}

        index: dict[str, _ChannelRepoInfo] = {}
        for url in url_tuple:
            info = self._json_path_to_repo_info(url, jsons[url])
            index[info.noauth_url] = info

        return index

    def _load_pkgs_cache(self, pkgs_dirs: PathsType) -> list[_ChannelRepoInfo]:
        repos = []
        subdir = next(s for s in self._subdirs if s != "noarch")
        for path in pkgs_dirs:
            path_as_url = path_to_url(path)
            package_cache_data = PackageCacheData(path)
            package_cache_data.load()
            arch = empty_repodata_dict(subdir, base_url=path_as_url)
            noarch = empty_repodata_dict("noarch", base_url=path_as_url)
            for record in package_cache_data.values():
                record: PackageCacheRecord
                if record.subdir not in self._subdirs:
                    continue
                record_data = dict(record.dump())
                for field in (
                    "sha256",
                    "track_features",
                    "license",
                    "size",
                    "url",
                    "noarch",
                    "platform",
                    "timestamp",
                ):
                    if field in record_data:
                        continue  # do not overwrite
                    value = getattr(record, field, None)
                    if value is not None:
                        record_data[field] = value
                key = "packages" if record.fn.endswith(".tar.bz2") else "packages.conda"
                if record.noarch:
                    noarch[key][record.fn] = record_data
                else:
                    arch[key][record.fn] = record_data
            for subdir_name, repodata in (("noarch", noarch), (subdir, arch)):
                with NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                    f.write(json_dump(repodata))
                repos.append(
                    _ChannelRepoInfo(
                        repo=rattler.SparseRepoData(
                            rattler.Channel(path_as_url),
                            subdir_name,
                            f.name,
                        ),
                        channel=Channel(path_as_url),
                        full_url=path_as_url,
                        noauth_url=path_as_url,
                        local_json=f.name,
                    )
                )
                self._unlink_on_del.append(Path(f.name))
        return repos

    def search(self, spec: str | MatchSpec) -> Iterable[PackageRecord]:
        spec = rattler.MatchSpec(str(spec))
        for info in self._index.values():
            for record in info.repo.load_matching_records([spec]):
                yield rattler_record_to_conda_record(record)

    @property
    def _package_format(self) -> rattler.PackageFormatSelection:
        return (
            rattler.PackageFormatSelection.ONLY_TAR_BZ2
            if context.use_only_tar_bz2
            else rattler.PackageFormatSelection.PREFER_CONDA_WITH_WHL
        )

    def __del__(self):
        if self._unlink_on_del:
            for info in self._index.values():
                info.repo.close()
            self._index.clear()
        for path in self._unlink_on_del:
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                print(exc, file=sys.stderr)  # Debug
                print(self._index, file=sys.stderr)  # Debug
