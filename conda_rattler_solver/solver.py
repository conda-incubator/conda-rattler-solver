from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from functools import cache
from inspect import stack
from itertools import chain
from typing import TYPE_CHECKING

import rattler
from conda.base.constants import REPODATA_FN, ChannelPriority
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import time_recorder
from conda.core.solve import Solver
from conda.exceptions import PackagesNotFoundError
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.reporters import get_spinner
from rattler import __version__ as rattler_version
from rattler.exceptions import SolverError as RattlerSolverError

from . import __version__
from .exceptions import RattlerUnsatisfiableError
from .index import RattlerIndexHelper
from .state import SolverInputState, SolverOutputState
from .utils import (
    conda_match_spec_to_rattler_match_spec,
    conda_prefix_record_to_rattler_prefix_record,
    fix_version_field_for_conda_build,
    maybe_ignore_current_repodata,
    notify_conda_outdated,
    rattler_record_to_conda_record,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from boltons.setutils import IndexedSet
    from conda.auxlib import _Null
    from conda.base.constants import (
        DepsModifier,
        UpdateModifier,
    )
    from conda.common.path import PathType
    from conda.models.records import PackageRecord

log = logging.getLogger(f"conda.{__name__}")


class RattlerSolver(Solver):
    MAX_SOLVER_ATTEMPTS_CAP = 10
    _uses_ssc = False

    @staticmethod
    @cache
    def user_agent() -> str:
        """
        Expose this identifier to allow conda to extend its user agent if required
        """
        return f"conda-rattler-solver/{__version__} py-rattler/{rattler_version}"

    def __init__(
        self,
        prefix: PathType,
        channels: Iterable[Channel | str],
        subdirs: Iterable[str] = (),
        specs_to_add: Iterable[MatchSpec | str] = (),
        specs_to_remove: Iterable[MatchSpec | str] = (),
        repodata_fn: str = REPODATA_FN,
        command: str | _Null = NULL,
    ):
        if specs_to_add and specs_to_remove:
            raise ValueError(
                "Only one of `specs_to_add` and `specs_to_remove` can be set at a time"
            )
        if specs_to_remove and command is NULL:
            command = "remove"

        self._unmerged_specs_to_add = frozenset(MatchSpec(spec) for spec in specs_to_add)
        super().__init__(
            os.fspath(prefix),
            channels,
            subdirs=subdirs,
            specs_to_add=specs_to_add,
            specs_to_remove=specs_to_remove,
            repodata_fn=repodata_fn,
            command=command,
        )
        if self.subdirs is NULL or not self.subdirs:
            self.subdirs = context.subdirs
        if "noarch" not in self.subdirs:
            # Problem: Conda build generates a custom index which happens to "forget" about
            # noarch on purpose when creating the build/host environments, since it merges
            # both as if they were all in the native subdir. This causes package-not-found
            # errors because we are not using the patched index.
            # Fix: just add noarch to subdirs because it should always be there anyway.
            self.subdirs = (*self.subdirs, "noarch")

        self._repodata_fn = maybe_ignore_current_repodata(self._repodata_fn)

    def solve_final_state(
        self,
        update_modifier: UpdateModifier | _Null = NULL,
        deps_modifier: DepsModifier | _Null = NULL,
        prune: bool | _Null = NULL,
        ignore_pinned: bool | _Null = NULL,
        force_remove: bool | _Null = NULL,
        should_retry_solve: bool = False,
    ) -> IndexedSet[PackageRecord]:
        in_state = SolverInputState(
            prefix=self.prefix,
            requested=self.specs_to_add or self.specs_to_remove,
            update_modifier=update_modifier,
            deps_modifier=deps_modifier,
            prune=prune,
            ignore_pinned=ignore_pinned,
            force_remove=force_remove,
            command=self._command,
        )

        out_state = SolverOutputState(solver_input_state=in_state)

        # These tasks do _not_ require a solver...
        # TODO: Abstract away in the base class?
        none_or_final_state = out_state.early_exit()
        if none_or_final_state is not None:
            return none_or_final_state

        channels = self._collect_channel_list(in_state)
        conda_build_channels = self._collect_channels_subdirs_from_conda_build(seen=set(channels))
        with get_spinner(
            self._collect_all_metadata_spinner_message(channels, conda_build_channels),
        ):
            index = self._collect_all_metadata(
                channels=channels,
                conda_build_channels=conda_build_channels,
                subdirs=self.subdirs,
            )
            out_state.check_for_pin_conflicts(index)

        with get_spinner(
            self._solving_loop_spinner_message(),
        ):
            # This function will copy and mutate `out_state`
            # Make sure we get the latest copy to return the correct solution below
            out_state = self._solving_loop(in_state, out_state, index)
            self.neutered_specs = tuple(out_state.neutered.values())
            solution = out_state.current_solution

        # Check whether conda can be updated; this is normally done in .solve_for_diff()
        # but we are doing it now so we can reuse in_state and friends
        self._notify_conda_outdated(None, self.prefix, index, solution)

        return solution

    # region Metadata collection

    def _collect_all_metadata_spinner_message(
        self,
        channels: Iterable[Channel],
        conda_build_channels: Iterable[Channel | str] = (),
    ) -> str:
        if self._called_from_conda_build():
            msg = "Reloading output folder"
            if conda_build_channels:
                names = list(
                    dict.fromkeys([Channel(c).canonical_name for c in conda_build_channels])
                )
                msg += f" ({', '.join(names)})"
            return msg

        canonical_names = list(dict.fromkeys([c.canonical_name for c in channels]))
        if len(canonical_names) > 1:
            canonical_names_dashed = "\n - ".join(canonical_names)
            return (
                f"Channels:\n"
                f" - {canonical_names_dashed}\n"
                f"Platform: {context.subdir}\n"
                f"Target prefix: {self.prefix}\n"
                f"Collecting package metadata ({self._repodata_fn})"
            )
        else:
            return (
                f"Channel: {''.join(canonical_names)}\n"
                f"Platform: {context.subdir}\n"
                f"Target prefix: {self.prefix}\n"
                f"Collecting package metadata ({self._repodata_fn})"
            )

    def _collect_channel_list(self, in_state: SolverInputState) -> list[Channel]:
        # Aggregate channels and subdirs
        deduped_channels = {}
        for channel in chain(
            self.channels, in_state.channels_from_specs(), in_state.maybe_free_channel()
        ):
            if channel_platform := getattr(channel, "platform", None):
                if channel_platform not in self.subdirs:
                    log.info(
                        "Channel %s defines platform %s which is not part of subdirs=%s. "
                        "Ignoring platform attribute...",
                        channel,
                        channel_platform,
                        self.subdirs,
                    )
                # Remove 'Channel.platform' to avoid missing subdirs. Channel.urls() will ignore
                # our explicitly passed subdirs if .platform is defined!
                channel = Channel(**{k: v for k, v in channel.dump().items() if k != "platform"})
            deduped_channels[channel] = None
        return list(deduped_channels)

    def _collect_channels_subdirs_from_conda_build(
        self,
        seen: set[Channel] | None = None,
    ) -> list[Channel]:
        if self._called_from_conda_build():
            seen = seen or set()
            # We need to recover the local dirs (conda-build's local, output_folder, etc)
            # from the index. This is a bit of a hack, but it works.
            conda_build_channels = {}
            for record in self._index or {}:
                if record.channel.scheme == "file":
                    # Remove 'Channel.platform' to avoid missing subdirs. Channel.urls()
                    # will ignore our explicitly passed subdirs if .platform is defined!
                    channel = Channel(
                        **{k: v for k, v in record.channel.dump().items() if k != "platform"}
                    )
                    if channel not in seen:
                        conda_build_channels.setdefault(channel)
            return list(conda_build_channels)
        return []

    @time_recorder(module_name=__name__)
    def _collect_all_metadata(
        self,
        channels: Iterable[Channel],
        conda_build_channels: Iterable[Channel],
        subdirs: Iterable[str],
    ) -> RattlerIndexHelper:
        index = RattlerIndexHelper(
            channels=[*conda_build_channels, *channels],
            subdirs=subdirs,
            repodata_fn=self._repodata_fn,
            pkgs_dirs=context.pkgs_dirs if context.offline else (),
        )
        for channel in conda_build_channels:
            index.reload_channel(channel)
        return index

    # endregion
    # region Solve

    def _solving_loop_spinner_message(self) -> str:
        """This shouldn't be our responsibility, but the CLI / app's..."""
        prefix_name = os.path.basename(self.prefix)
        if self._called_from_conda_build():
            if "_env" in prefix_name:
                env_name = "_".join(prefix_name.split("_")[:3])
                return f"Solving environment ({env_name})"
            else:
                # https://github.com/conda/conda-build/blob/e0884b626a/conda_build/environ.py#L1035-L1036
                return "Getting pinned dependencies"
        return "Solving environment"

    def _solving_loop(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: RattlerIndexHelper,
    ) -> SolverOutputState:
        solution = None
        out_state.check_for_pin_conflicts(index)
        # Give some
        if n_installed := len(in_state.installed):
            max_attempts = min(n_installed, self.MAX_SOLVER_ATTEMPTS_CAP)
        else:
            max_attempts = 1
        for attempt in range(1, max_attempts + 1):
            log.debug("Starting solver attempt %s", attempt)
            solution = self._solve_attempt(in_state, out_state, index, attempt=attempt)
            if not isinstance(solution, Exception):  # Found a solution, stop trying
                break
            out_state = SolverOutputState(
                solver_input_state=in_state,
                records=dict(out_state.records),
                for_history=dict(out_state.for_history),
                neutered=dict(out_state.neutered),
                conflicts=dict(out_state.conflicts),
                pins=dict(out_state.pins),
            )
        else:
            # Didn't find a solution yet, let's unfreeze everything
            out_state.conflicts.update(
                {
                    name: record.to_match_spec()
                    for name, record in in_state.installed.items()
                    if not record.is_unmanageable
                }
            )
            solution = self._solve_attempt(
                in_state,
                out_state,
                index,
                attempt=attempt + 1000,  # last attempt
            )
            if isinstance(solution, Exception) or solution is None:
                exc = RattlerUnsatisfiableError(solution or "Could not find solution")
                exc.allow_retry = False
                raise exc

        # We didn't fail? Nice, let's return the calculated state
        self._export_solved_records(solution, out_state)

        # Run post-solve tasks
        out_state.post_solve(solver=self)

        return out_state

    def _solve_attempt(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: RattlerIndexHelper,
        attempt: int = 0,
    ) -> RattlerSolverError | list[rattler.RepoDataRecord]:
        """
        The solver in rattler is declarative: you pass the things you want
        and it returns either a solution or an exception with the conflicts.
        There's no concept of 'remove'. Thus, we need to structure the requests
        a bit differently than in classic or libmamba.

        The solver API offers these categories:

        - specs: MatchSpecs to _install_.
        - constraints: Additional conditions (as MatchSpecs) to meet,
          _if_ the package mentioned ends up in the install list.
        - locked_packages: Preferred records (useful to minimize number of updates and respect
          the installed packages, history or lockfiles).
        - pinned_packages: Records that MUST be present if needed. It will not allow any other
          variant.
        - virtual_packages: Details of the system.
        """
        solve_kwargs = self._collect_specs(in_state, out_state)
        if log.isEnabledFor(logging.DEBUG):
            dumped = json.dumps(solve_kwargs, indent=2, default=str, sort_keys=True)
            log.debug("Solver input:\n%s", dumped)
        try:
            solution = asyncio.run(
                rattler.solve_with_sparse_repodata(
                    **solve_kwargs,
                    virtual_packages=self._rattler_virtual_packages(in_state),
                    sparse_repodata=[info.repo for info in index._index.values()],
                    channel_priority=(
                        rattler.ChannelPriority.Strict
                        if context.channel_priority == ChannelPriority.STRICT
                        else rattler.ChannelPriority.Disabled
                    ),
                    strategy="highest",
                    use_only_tar_bz2=context.use_only_tar_bz2,
                )
            )
        except RattlerSolverError as exc:
            self._maybe_raise_for_problems(str(exc), out_state)
            return exc
        else:
            out_state.conflicts.clear()
            return solution

    def _collect_specs(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
    ) -> dict[str, list[rattler.MatchSpec] | list[rattler.PackageRecord]]:
        if in_state.is_removing:
            return self._collect_specs_for_remove(in_state, out_state)
        elif self._called_from_conda_build():
            return self._collect_specs_for_conda_build(in_state)

        specs: list[rattler.MatchSpec] = []
        constraints: list[rattler.MatchSpec] = []
        locked_packages: list[rattler.PackageRecord] = []
        pinned_packages: list[rattler.PackageRecord] = []

        # Protect history and aggressive updates from being uninstalled if possible. From libsolv
        # docs: "The matching installed packages are considered to be installed by a user, thus not
        # installed to fulfill some dependency. This is needed input for the calculation of
        # unneeded packages for jobs that have the SOLVER_CLEANDEPS flag set."
        user_installed = {
            pkg
            for pkg in (
                *in_state.history,
                *in_state.aggressive_updates,
                *in_state.pinned,
                *in_state.do_not_remove,
            )
            if pkg in in_state.installed
        }

        # Fast-track python version changes (Part 1/2)
        # ## When the Python version changes, this implies all packages depending on
        # ## python will be reinstalled too. This can mean that we'll have to try for every
        # ## installed package to result in a conflict before we get to actually solve everything
        # ## A workaround is to let all non-noarch python-depending specs to "float" by marking
        # ## them as a conflict preemptively
        python_version_might_change = False
        installed_python = in_state.installed.get("python")
        to_be_installed_python = out_state.specs.get("python")
        if installed_python and to_be_installed_python:
            python_version_might_change = not to_be_installed_python.match(installed_python)

        # TODO: Make in_state.requested a dict[str, list[MatchSpec]]
        # This makes tests/core/test_solve.py::test_globstr_matchspec_compatible
        # and test_globstr_matchspec_non_compatible pass
        requested_specs = defaultdict(list)
        for spec in self._unmerged_specs_to_add:
            requested_specs[spec.name].append(spec)

        for name in out_state.specs:
            if "*" in name:
                continue

            installed: PackageRecord = in_state.installed.get(name)
            requested: list[MatchSpec] = requested_specs.get(name)
            history: MatchSpec = in_state.history.get(name)
            pinned: MatchSpec = in_state.pinned.get(name)
            conflicting: MatchSpec = out_state.conflicts.get(name)

            if (
                name in user_installed
                and not in_state.prune
                and not conflicting
                and not requested
            ):
                locked_packages.append(installed)

            if pinned and not pinned.is_name_only_spec:
                constraints.append(pinned)

            if requested:
                specs.extend(requested)
            elif name in in_state.always_update and not conflicting:
                constraints.append(f"{name}>={installed.version}")
            # These specs are "implicit"; the solver logic massages them for better UX
            # as long as they don't cause trouble
            elif in_state.prune:
                continue
            elif name == "python" and installed and not pinned:
                pyver = ".".join(installed.version.split(".")[:2])
                constraints.append(f"python {pyver}.*")
            elif history:
                if conflicting and history.strictness == 3:
                    # relax name-version-build (strictness=3) history specs that cause conflicts
                    # this is called neutering and makes test_neutering_of_historic_specs pass
                    version = str(history.version or "")
                    if version.startswith("=="):
                        spec_str = f"{name} {version[2:]}"
                    elif version.startswith(("!=", ">", "<")):
                        spec_str = f"{name} {version}"
                    elif version:
                        spec_str = f"{name} {version}.*"
                    else:
                        spec_str = name
                    specs.append(spec_str)
                else:
                    specs.append(history)
            elif installed and not conflicting:
                # we freeze everything else as installed
                lock = in_state.update_modifier.FREEZE_INSTALLED
                if pinned and pinned.is_name_only_spec:
                    # name-only pins are treated as locks when installed
                    lock = True
                if python_version_might_change and installed.noarch is None:
                    for dep in installed.depends:
                        if MatchSpec(dep).name in ("python", "python_abi"):
                            lock = False
                            break
                if lock:
                    pinned_packages.append(installed)
                else:
                    specs.append(installed.name)
                    if installed not in locked_packages:
                        locked_packages.append(installed)
        return {
            "specs": [conda_match_spec_to_rattler_match_spec(spec) for spec in specs],
            "constraints": [conda_match_spec_to_rattler_match_spec(spec) for spec in constraints],
            "locked_packages": [
                conda_prefix_record_to_rattler_prefix_record(record) for record in locked_packages
            ],
            "pinned_packages": [
                conda_prefix_record_to_rattler_prefix_record(record) for record in pinned_packages
            ],
        }

    def _collect_specs_for_remove(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
    ) -> dict[str, list[rattler.MatchSpec] | list[rattler.PackageRecord]]:
        specs: list[rattler.MatchSpec] = []
        constraints: list[rattler.MatchSpec] = []
        locked_packages: list[rattler.PackageRecord] = []
        pinned_packages: list[rattler.PackageRecord] = []

        # conda remove allows globbed names; make sure we don't install those!
        remove = set()
        for requested_name, requested_spec in in_state.requested.items():
            if "*" in requested_name:
                for installed_name, installed_record in in_state.installed.items():
                    if requested_spec.match(installed_record):
                        remove.add(installed_name)
            else:
                remove.add(requested_name)

        for name in out_state.specs:
            if "*" in name:
                continue

            installed: PackageRecord = in_state.installed.get(name)
            history: MatchSpec = in_state.history.get(name)
            pinned: MatchSpec = in_state.pinned.get(name)
            conflicting: MatchSpec = out_state.conflicts.get(name)

            if name in remove:
                constraints.append(f"{name}<0.0.0dev0")
                continue
            if pinned:
                if pinned.is_name_only_spec and installed:
                    pinned_packages.append(installed)
                else:
                    constraints.append(pinned)
            if installed:
                if name in in_state.aggressive_updates:
                    specs.append(name)
                elif not conflicting:
                    if (
                        history
                    ):  # TODO: Do this even if not installed (e.g. force removed previously?)
                        specs.append(history)
                    locked_packages.append(installed)

        return {
            "specs": [conda_match_spec_to_rattler_match_spec(spec) for spec in specs],
            "constraints": [conda_match_spec_to_rattler_match_spec(spec) for spec in constraints],
            "locked_packages": [
                conda_prefix_record_to_rattler_prefix_record(record) for record in locked_packages
            ],
            "pinned_packages": [
                conda_prefix_record_to_rattler_prefix_record(record) for record in pinned_packages
            ],
        }

    # endregion

    # region Error reporting

    def _maybe_raise_for_problems(self, problems: str, out_state: SolverOutputState):
        unsatisfiable = {}
        not_found = {}
        for line in problems.splitlines():
            for char in "─│└├":
                line = line.replace(char, "")
            line = line.strip()
            if line.startswith("Cannot solve the request because of:"):
                line = line.split(":", 1)[1]
            words = line.split()
            if "is locked, but another version is required as reported above" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1]}")
            elif "which cannot be installed because there are no viable options" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1].strip(',')}")
            elif "cannot be installed because there are no viable options" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1]}")
            elif "the constraint" in line and "cannot be fulfilled" in line:
                unsatisfiable[words[2]] = MatchSpec(" ".join(words[2:-3]))
            elif (
                "can be installed with any of the following options" in line
                and "which" not in line
            ):
                position = line.index(" can be installed with")
                unsatisfiable[words[0]] = MatchSpec(line[:position])
            elif "No candidates were found for" in line:
                position = line.index("No candidates were found for ")
                position += len("No candidates were found for ")
                spec = line[position:].rstrip(".")
                not_found[spec.split()[0]] = MatchSpec(spec)
        if not unsatisfiable and not_found:
            log.debug(
                "Inferred PackagesNotFoundError %s from conflicts:\n%s",
                tuple(not_found.keys()),
                problems,
            )
            # This is not a conflict, but a missing package in the channel
            exc = PackagesNotFoundError(tuple(not_found.values()), tuple(self.channels))
            exc.allow_retry = False
            raise exc

        previous = out_state.conflicts or {}
        previous_set = set(previous.values())
        current_set = set(unsatisfiable.values())

        diff = current_set.difference(previous_set)
        if len(diff) > 1 and "python" in unsatisfiable:
            # Only report python as conflict if it's the only conflict reported
            # This helps us prioritize neutering for other dependencies first
            unsatisfiable.pop("python")

        if (previous and (previous_set == current_set)) or len(diff) >= 10:
            # We have same or more (up to 10) unsatisfiable now! Abort to avoid recursion
            exc = RattlerUnsatisfiableError(problems)
            # do not allow conda.cli.install to try more things
            exc.allow_retry = False
            raise exc

        log.debug("Attempt failed with %s conflicts:\n%s", len(unsatisfiable), problems)
        out_state.conflicts.update(unsatisfiable)

    def _maybe_raise_for_conda_build(
        self,
        conflicting_specs: Mapping[str, MatchSpec],
        message: str = None,
    ) -> None:
        # TODO: Remove this hack for conda-build compatibility >_<
        # conda-build expects a slightly different exception format
        # good news is that we don't need to retry much, because all
        # conda-build envs are fresh - if we found a conflict, we report
        # right away to let conda build handle it
        if not self._called_from_conda_build():
            return
        if not conflicting_specs:
            return
        from ._conda_build_exceptions import ExplainedDependencyNeedsBuildingError

        # the patched index should contain the arch we are building this env for
        # if the index is empty, we default to whatever platform we are running on
        subdir = next((subdir for subdir in self.subdirs if subdir != "noarch"), context.subdir)
        exc = ExplainedDependencyNeedsBuildingError(
            packages=list(conflicting_specs.keys()),
            matchspecs=list(conflicting_specs.values()),
            subdir=subdir,
            explanation=message,
        )
        raise exc

    def _export_solved_records(self, records, out_state):
        out_state.records.clear()
        for rattler_record in records:
            conda_record = rattler_record_to_conda_record(rattler_record)
            out_state.records[conda_record.name] = conda_record

    # endregion
    # region Converters & Checkers

    def _rattler_virtual_packages(
        self, in_state: SolverInputState
    ) -> list[rattler.GenericVirtualPackage]:
        return [
            rattler.GenericVirtualPackage(
                rattler.PackageName(pkg.name),
                rattler.Version(pkg.version),
                pkg.build,
            )
            for pkg in in_state.virtual.values()
        ]

    def _called_from_conda_build(self) -> bool:
        """
        conda build calls the solver via `conda.plan.install_actions`, which
        overrides Solver._index (populated in the classic solver, but empty for us)
        with a custom index. We can use this to detect whether conda build is in use
        and apply some compatibility fixes.
        """
        return (
            # conda_build.environ.get_install_actions will always pass a custom 'index'
            # which conda.plan.install_actions uses to override our null Solver._index
            getattr(self, "_index", None)
            # Is conda build in use? In that case, it should have been imported
            and "conda_build" in sys.modules
            # Confirm conda_build.environ's 'get_install_actions' and conda.plan's
            # 'install_actions' are in the call stack. We don't check order or
            # contiguousness, but what are the chances at this point...?
            # frame[3] contains the name of the function in that frame of the stack
            and {"install_actions", "get_install_actions"} <= {frame[3] for frame in stack()}
        )

    def _collect_specs_for_conda_build(
        self,
        in_state: SolverInputState,
    ) -> dict[str, list[rattler.MatchSpec] | list[rattler.PackageRecord]]:
        specs = []
        for name, spec in in_state.requested.items():
            if name.startswith("__"):
                continue
            spec = fix_version_field_for_conda_build(spec)
            specs.append(spec.conda_build_form())

        return {
            "specs": [conda_match_spec_to_rattler_match_spec(spec) for spec in specs],
            "constraints": [],
            "locked_packages": [],
            "pinned_packages": [],
        }

    def _notify_conda_outdated(self, link_precs, *args, **kwargs):
        notify_conda_outdated(*args, **kwargs)
