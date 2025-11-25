from __future__ import annotations

import json
import os
import re
import sys
from contextlib import suppress
from enum import Enum
from logging import getLogger
from textwrap import dedent
from typing import TYPE_CHECKING

import rattler
from conda import __version__ as _conda_version
from conda.base.constants import KNOWN_SUBDIRS, REPODATA_FN, UNKNOWN_CHANNEL
from conda.base.context import context
from conda.common.path import paths_equal
from conda.exceptions import InvalidMatchSpec, PackagesNotFoundError
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord, PrefixRecord
from conda.models.version import VersionOrder

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from conda.common.path import PathType

    from .index import RattlerIndexHelper

log = getLogger(f"conda.{__name__}")


def _hash_to_str(bytes_or_str: bytes | str | None) -> None | str:
    if not bytes_or_str:
        return None
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.hex()
    return bytes_or_str.lower()


def rattler_record_to_conda_record(record: rattler.PackageRecord) -> PackageRecord:
    if timestamp := record.timestamp:
        timestamp = int(timestamp.timestamp() * 1000)
    else:
        timestamp = 0

    if record.noarch.none:
        noarch = None
    elif record.noarch.python:
        noarch = "python"
    elif record.noarch.generic:
        noarch = "generic"
    else:
        raise ValueError(f"Unknown noarch type: {record.noarch}")

    if record.channel:
        if record.channel.endswith(("noarch", *KNOWN_SUBDIRS)):
            channel_url = record.channel
        elif record.subdir:
            channel_url = f"{record.channel}/{record.subdir}"
        else:
            channel_url = record.channel
    else:
        channel_url = ""

    return PackageRecord(
        name=record.name.source,
        version=str(record.version),
        build=record.build,
        build_number=record.build_number,
        channel=channel_url,
        subdir=record.subdir,
        fn=record.file_name,
        md5=_hash_to_str(record.md5),
        legacy_bz2_md5=_hash_to_str(record.legacy_bz2_md5),
        legacy_bz2_size=record.legacy_bz2_size,
        url=record.url,
        sha256=_hash_to_str(record.sha256),
        arch=record.arch,
        platform=str(record.platform or "") or None,
        depends=record.depends or (),
        constrains=record.constrains or (),
        track_features=record.track_features or (),
        features=record.features or (),
        noarch=noarch,
        # preferred_env=record.preferred_env,
        license=record.license,
        license_family=record.license_family,
        # package_type=record.package_type,
        timestamp=timestamp,
        # date=record.date,
        size=record.size or 0,
        python_site_packages_path=record.python_site_packages_path,
    )


class FakeRattlerLinkType(Enum):
    # directory is not a link type, and copy is not a path type
    # LinkType is still probably the best name here
    hardlink = "hardlink"
    softlink = "softlink"
    copy = "copy"
    directory = "directory"


def conda_prefix_record_to_rattler_prefix_record(
    record: PrefixRecord,
) -> rattler.PrefixRecord:
    if platform := record.get("platform"):
        platform = platform.value
    if noarch := record.get("noarch"):
        noarch = rattler.NoArchType(noarch.value)
    package_record = rattler.PackageRecord(
        name=record.name,
        version=record.version,
        build=record.build,
        build_number=record.build_number,
        subdir=record.subdir,
        arch=record.get("arch"),
        platform=platform,
        noarch=noarch,
        depends=record.get("depends"),
        constrains=record.get("constrains"),
        sha256=bytes.fromhex(record.get("sha256") or "") or None,
        md5=bytes.fromhex(record.get("md5", "") or "") or None,
        size=record.get("size"),
        features=record.get("features") or None,
        legacy_bz2_md5=bytes.fromhex(record.get("legacy_bz2_md5", "") or "") or None,
        legacy_bz2_size=bytes.fromhex(record.get("legacy_bz2_size", "") or "") or None,
        license=record.get("license"),
        license_family=record.get("license_family"),
        python_site_packages_path=record.get("python_site_packages_path"),
    )
    repodata_record = rattler.RepoDataRecord(
        package_record=package_record,
        file_name=record.fn,
        url=record.url,
        channel=record.channel.base_url,
    )
    paths_data = rattler.PrefixPaths()
    if conda_paths_data := record.get("paths_data"):
        path_entries = []
        for path in conda_paths_data.paths:
            path_type = str(path.path_type)
            kwargs = {
                "relative_path": path.path,
                "path_type": rattler.PrefixPathType(path_type),
                "prefix_placeholder": getattr(path, "prefix_placeholder", None),
                "sha256": bytes.fromhex(getattr(path, "sha256", "")) or None,
                "sha256_in_prefix": bytes.fromhex(getattr(path, "sha256_in_prefix", "")) or None,
                "size_in_bytes": getattr(path, "size_in_bytes", None),
            }
            if file_mode := str(getattr(path, "file_mode", "")):
                kwargs["file_mode"] = rattler.FileMode(file_mode)
                path_entries.append(rattler.PrefixPathsEntry(**kwargs))
        paths_data.paths = path_entries
    if conda_link := record.get("link"):
        link_type = FakeRattlerLinkType(str(conda_link.type))
        link = rattler.Link(path=conda_link.source, type=link_type)
    else:
        link = None
    return rattler.PrefixRecord(
        repodata_record=repodata_record,
        paths_data=paths_data,
        link=link,
        package_tarball_full_path=record.get("package_tarball_full_path"),
        extracted_package_dir=record.get("extracted_package_dir"),
        requested_spec=record.get("requested_spec"),
        files=record.files,
    )


def conda_match_spec_to_rattler_match_spec(spec: MatchSpec) -> rattler.MatchSpec:
    match_spec = MatchSpec(spec)
    if os.sep in match_spec.name or "/" in match_spec.name:
        raise InvalidMatchSpec(match_spec, "Cannot contain slashes.")
    return rattler.MatchSpec(str(match_spec).rstrip("=").replace("=[", "["))


def empty_repodata_dict(subdir: str, **info_kwargs) -> dict[str, Any]:
    return {
        "info": {
            "subdir": subdir,
            **info_kwargs,
        },
        "packages": {},
        "packages.conda": {},
    }


def maybe_ignore_current_repodata(repodata_fn) -> str:
    is_repodata_fn_set = False
    for config in context.collect_all().values():
        for key, value in config.items():
            if key == "repodata_fns" and value:
                is_repodata_fn_set = True
                break
    if repodata_fn == "current_repodata.json" and not is_repodata_fn_set:
        log.debug(
            "Ignoring repodata_fn='current_repodata.json', defaulting to %s",
            REPODATA_FN,
        )
        return REPODATA_FN
    return repodata_fn


def notify_conda_outdated(
    prefix: PathType | None = None,
    index: RattlerIndexHelper | None = None,
    final_state: Iterable[PackageRecord] | None = None,
) -> None:
    """
    We are overriding the base class implementation, which gets called in
    Solver.solve_for_diff() once 'link_precs' is available. However, we
    are going to call it before (in .solve_final_state(), right after the solve).
    That way we can reuse the IndexHelper and SolverOutputState instances we have
    around, which contains the channel and env information we need, before losing them.
    """
    if prefix is None and index is None and final_state is None:
        # The parent class 'Solver.solve_for_diff()' method will call this method again
        # with only 'link_precs' as the argument, because that's the original method signature.
        # We have added two optional kwargs (index and final_state) so we can call this method
        # earlier, in .solve_final_state(), while we still have access to the index helper
        # (which allows us to query the available packages in the channels quickly, without
        # reloading the channels with conda) and the final_state (which gives the list of
        # packages to be installed). So, if both index and final_state are None, we return
        # because that means that the method is being called from .solve_for_diff() and at
        # that point we will have already called it from .solve_for_state().
        return
    if not context.notify_outdated_conda or context.quiet:
        # This check can be silenced with a specific option in the context or in quiet mode
        return

    # manually check base prefix since `PrefixData(...).get("conda", None) is expensive
    # once prefix data is lazy this might be a different situation
    current_conda_prefix_rec = None
    conda_meta_prefix_directory = os.path.join(context.conda_prefix, "conda-meta")
    with suppress(OSError, ValueError):
        if os.path.lexists(conda_meta_prefix_directory):
            for entry in os.scandir(conda_meta_prefix_directory):
                if (
                    entry.is_file()
                    and entry.name.endswith(".json")
                    and entry.name.rsplit("-", 2)[0] == "conda"
                ):
                    with open(entry.path) as f:
                        current_conda_prefix_rec = PrefixRecord(**json.loads(f.read()))
                    break
    if not current_conda_prefix_rec:
        # We are checking whether conda can be found in the environment conda is
        # running from. Unless something is really wrong, this should never happen.
        return

    channel_name = current_conda_prefix_rec.channel.canonical_name
    if channel_name in (UNKNOWN_CHANNEL, "@", "<develop>", "pypi"):
        channel_name = "defaults"

    # only check the loaded index if it contains the channel conda should come from
    # otherwise ignore
    index_channels = {getattr(chn, "canonical_name", chn) for chn in index.channels}
    if channel_name not in index_channels:
        return

    # we only want to check if a newer conda is available in the channel we installed it from
    conda_newer_str = f"{channel_name}::conda>{_conda_version}"
    conda_newer_spec = MatchSpec(conda_newer_str)

    # if target prefix is the same conda is running from
    # maybe the solution we are proposing already contains
    # an updated conda! in that case, we don't need to check further
    if paths_equal(prefix, context.conda_prefix):
        if any(conda_newer_spec.match(record) for record in final_state):
            return

    # check if the loaded index contains records that match a more recent conda version
    conda_newer_records = list(index.search(conda_newer_str))

    # print instructions to stderr if we found a newer conda
    if conda_newer_records:
        newest = max(conda_newer_records, key=lambda x: VersionOrder(x.version))
        print(
            dedent(
                f"""

                    ==> WARNING: A newer version of conda exists. <==
                        current version: {_conda_version}
                        latest version: {newest.version}

                    Please update conda by running

                        $ conda update -n base -c {channel_name} conda

                    """
            ),
            file=sys.stderr,
        )


def fix_version_field_for_conda_build(spec: MatchSpec) -> MatchSpec:
    """Fix taken from mambabuild"""
    if spec.version:
        only_dot_or_digit_re = re.compile(r"^[\d\.]+$")
        version_str = str(spec.version)
        if re.match(only_dot_or_digit_re, version_str):
            spec_fields = spec.conda_build_form().split()
            if version_str.count(".") <= 1:
                spec_fields[1] = version_str + ".*"
            else:
                spec_fields[1] = version_str + "*"
            return MatchSpec(" ".join(spec_fields))
    return spec


def compatible_specs(
    index: RattlerIndexHelper, specs: Iterable[MatchSpec], raise_not_found: bool = True
) -> bool:
    """
    Assess whether the given specs are compatible with each other.
    This is done by querying the index for each spec and taking the
    intersection of the results. If the intersection is empty, the
    specs are incompatible.

    If raise_not_found is True, a PackagesNotFoundError will be raised
    when one of the specs is not found. Otherwise, False will be returned
    because the intersection will be empty.
    """
    if not len(specs) >= 2:
        raise ValueError("Must specify at least two specs")

    matched = None
    for spec in specs:
        results = set(index.search(spec))
        if not results:
            if raise_not_found:
                exc = PackagesNotFoundError([spec], index.channels)
                exc.allow_retry = False
                raise exc
            return False
        if matched is None:
            # First spec, just set matched to the results
            matched = results
            continue
        # Take the intersection of the results
        matched &= results
        if not matched:
            return False

    return bool(matched)


class EnumAsBools:
    """
    Allows an Enum to be bool-evaluated with attribute access.

    >>> update_modifier = UpdateModifier("update_deps")
    >>> update_modifier_as_bools = EnumAsBools(update_modifier)
    >>> update_modifier == UpdateModifier.UPDATE_DEPS  # from this
        True
    >>> update_modidier_as_bools.UPDATE_DEPS  # to this
        True
    >>> update_modifier_as_bools.UPDATE_ALL
        False
    """

    def __init__(self, enum: Enum):
        self._enum = enum
        self._names = {v.name for v in self._enum.__class__.__members__.values()}

    def __getattr__(self, name: str) -> bool:
        if name in ("name", "value"):
            return getattr(self._enum, name)
        if name in self._names:
            return self._enum.name == name
        raise AttributeError(f"'{name}' is not a valid name for {self._enum.__class__.__name__}")

    def __eq__(self, obj: object) -> bool:
        return self._enum.__eq__(obj)

    def _dict(self) -> dict[str, bool]:
        return {name: self._enum.name == name for name in self._names}
