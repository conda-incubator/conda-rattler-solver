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
from conda.common.io import DummyExecutor, ThreadLimitedThreadPoolExecutor, time_recorder
from conda.common.url import path_to_url, remove_auth, split_anaconda_token
from conda.core.package_cache_data import PackageCacheData
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel

try:
    from conda.common.serialize.json import dumps as json_dump
except ImportError:
    from conda.common.serialize import json_dump

from .shards_subset import build_repodata_subset
from .utils import empty_repodata_dict, rattler_record_to_conda_record

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Self

    from conda.common.path import PathsType
    from conda.gateways.repodata import RepodataState
    from conda.models.match_spec import MatchSpec
    from conda.models.records import PackageCacheRecord, PackageRecord

    from .shards import ShardBase
    from .state import SolverInputState

log = logging.getLogger(f"conda.{__name__}")


@dataclass
class _ChannelRepoInfo:
    "A dataclass mapping conda Channels, rattler.SparseRepoData, URLs and JSON paths"

    channel: Channel | None
    repo: rattler.SparseRepoData
    full_url: str
    noauth_url: str
    local_json: str | None


def _is_sharded_repodata_enabled() -> bool:
    """
    Flag to see whether we should check for sharded repodata.
    """
    # NOTE: We may need to turn this into a conda setting, not just conda-libmamba-solver's
    return context.plugins.use_sharded_repodata is True


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

        self._channels = context.channels if channels is None else channels
        self._subdirs = context.subdirs if subdirs is None else subdirs
        self._repodata_fn = repodata_fn
        self._in_state = in_state

        self._index: dict[str, _ChannelRepoInfo] = {
            info.noauth_url: info for info in self._load_channels()
        }
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
        filter_: Callable | None = None,
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

    def _fetch_channel(self, url: str) -> tuple[str, os.PathLike, RepodataState]:
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
        json_path, state = subdir_data.repo_fetch.fetch_latest_path()

        return url, json_path, state

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
        repo = rattler.SparseRepoData(rattler_channel, channel.subdir or "noarch", json_path)
        return _ChannelRepoInfo(
            repo=repo,
            channel=channel,
            full_url=url,
            noauth_url=noauth_url,
            local_json=json_path,
        )

    def _channel_to_id(self, channel: Channel):
        channel_id = channel.canonical_name
        if channel_id in context.custom_multichannels:
            # In multichannels, the canonical name of a "subchannel" is the multichannel name
            # which makes it ambiguous for `channel::specs`. In those cases, take the channel
            # regular name; e.g. for repo.anaconda.com/pkgs/main, do not take defaults, but
            # pkgs/main instead.
            channel_id = channel.name
        return channel_id

    def _load_channels(
        self,
        urls_to_channel: dict[str, Channel] | None = None,
        try_solv: bool = True,
    ) -> list[_ChannelRepoInfo]:
        if urls_to_channel is None:
            urls_to_channel = self._channel_urls(self._subdirs, self._channels)

        urls_to_channel = self._encoded_urls_to_channels(urls_to_channel)

        # Prefer sharded repodata loading if it's enabled
        if self._in_state is not None and _is_sharded_repodata_enabled():
            # TODO: It may be better to directly pass channel objects without URL encoding

            # Channels are not loaded in order from shards. Once we load the shards,
            # reorder the list to match the intended channel order. If the channel
            # is not in the urls_to_channel, we'll add it to the back
            channel_repos_info = self._load_channel_repo_info_shards(urls_to_channel)
            urls_to_channel_as_list = list(urls_to_channel.keys())
            return sorted(
                channel_repos_info,
                key=lambda x: urls_to_channel_as_list.index(x.channel) or 0,
            )

        # Fallback to repodata.json loading
        return self._load_channel_repo_info_json(urls_to_channel, try_solv)

    def _fetch_repodata_jsons(self, urls: Iterable[str]) -> dict[str, tuple[str, RepodataState]]:
        Executor = (
            DummyExecutor
            if context.debug or context.repodata_threads == 1
            else partial(ThreadLimitedThreadPoolExecutor, max_workers=context.repodata_threads)
        )
        with Executor() as executor:
            return {
                url: (str(path), state)
                for (url, path, state) in executor.map(self._fetch_channel, urls)
            }

    @staticmethod
    def _encoded_urls_to_channels(urls_to_channel: dict[str, Channel]) -> dict[str, Channel]:
        """
        Return copy of urls_to_channel with %-encoded spaces.

        Usage: _encoded_urls_to_channels(_channel_urls(subdirs, channels))
        """
        # conda.common.url.path_to_url does not %-encode spaces
        encoded_urls_to_channel: dict[str, Channel] = {}
        for url, channel in urls_to_channel.items():
            if url.startswith("file://"):
                url = url.replace(" ", "%20")
            encoded_urls_to_channel[url] = channel
        return encoded_urls_to_channel

    def _load_channel_repo_info_shards(
        self, urls_to_channel: dict[str, Channel]
    ) -> list[_ChannelRepoInfo]:
        """
        Load repository information from sharded repodata cache.
        """
        # make a subset of possible dependencies
        root_packages = (*self._in_state.installed.keys(), *self._in_state.requested)
        channel_data = build_repodata_subset(root_packages, urls_to_channel)
        channel_repo_infos = self._load_channel_repo_info_shards_inner(channel_data)

        return channel_repo_infos

    @time_recorder(module_name=__name__)
    def _load_channel_repo_info_shards_inner(
        self, repodata_subset: dict[str, ShardBase]
    ) -> list[_ChannelRepoInfo]:
        """
        Load repository information from deserialized repodata.json-like
        structures.

        NOTE: This function initially converted all the raw repodata entries
        to libmamba PackageInfo objects. rattler does not support loading from its
        objects so we go through JSON on disk.
        """
        repos = []
        for channel_url, shardlike in repodata_subset.items():
            repodata = shardlike.build_repodata()
            repodata_tmp = NamedTemporaryFile(suffix=".json", delete=False)
            repodata_path = Path(repodata_tmp.name)
            self._unlink_on_del.append(repodata_path)
            repodata_path.write_text(json_dump(repodata, sort_keys=True))
            repos.append(self._json_path_to_repo_info(channel_url, repodata_tmp.name))

        return repos

    def _channel_to_id(self, channel: Channel):
        channel_id = channel.canonical_name
        if channel_id in context.custom_multichannels:
            # In multichannels, the canonical name of a "subchannel" is the multichannel name
            # which makes it ambiguous for `channel::specs`. In those cases, take the channel
            # regular name; e.g. for repo.anaconda.com/pkgs/main, do not take defaults, but
            # pkgs/main instead.
            channel_id = channel.name
        return channel_id

    def _load_channel_repo_info_json(
        self, urls_to_channel: dict[str, Channel], try_solv: bool
    ) -> list[_ChannelRepoInfo]:
        """
        Load repository information from repodata.json files.
        """
        urls_to_json_path_and_state = self._fetch_repodata_jsons(tuple(urls_to_channel.keys()))

        channel_repo_infos = []
        for url_w_cred, (json_path, _) in urls_to_json_path_and_state.items():
            channel_repo_infos.append(self._json_path_to_repo_info(url_w_cred, json_path))
        return channel_repo_infos

    @staticmethod
    def _channel_urls(subdirs: Iterable[str], channels: list[Channel]) -> dict[str, Channel]:
        "Map authenticated URLs to channel objects."
        # class method for testing etc.
        urls = {}
        seen_noauth = set()
        channels_with_subdirs = []
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
            if noauth_urls != auth_urls:  # authed channel always takes precedence
                urls.update({url: channel for url in auth_urls})
                seen_noauth.update(noauth_urls)
                continue
            # at this point, we are handling an unauthed channel; in some edge cases,
            # an auth'd variant of the same channel might already be present in `urls`.
            # we only add them if we haven't seen them yet
            for url in noauth_urls:
                if url not in seen_noauth:
                    urls[url] = channel
                    seen_noauth.add(url)
        return urls

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
            else rattler.PackageFormatSelection.PREFER_CONDA
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
