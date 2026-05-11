# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
import json
import os
import shutil
import time
from pathlib import Path

import pytest
from conda.base.context import context, reset_context
from conda.core.subdir_data import SubdirData
from conda.gateways.logging import initialize_logging
from conda.models.channel import Channel
from conda_rattler_solver.state import SolverInputState

from conda_rattler_solver.index import RattlerIndexHelper, _is_sharded_repodata_enabled

initialize_logging()
DATA = Path(__file__).parent / "data"

CONDA_FORGE_WITH_SHARDS = "conda-forge"


def test_given_channels(monkeypatch: pytest.MonkeyPatch, tmp_path: os.PathLike):
    monkeypatch.setenv("CONDA_PKGS_DIRS", str(tmp_path))
    reset_context()
    rattler_index = RattlerIndexHelper.from_platform_aware_channel(
        channel=Channel("conda-test/noarch")
    )
    assert len(rattler_index._index) == 1

    conda_index = SubdirData(Channel("conda-test/noarch"))
    conda_index.load()

    assert rattler_index.n_packages() == len(tuple(conda_index.iter_records()))


@pytest.mark.parametrize(
    "only_tar_bz2",
    (
        pytest.param("1", id="CONDA_USE_ONLY_TAR_BZ2=true"),
        pytest.param("", id="CONDA_USE_ONLY_TAR_BZ2=false"),
    ),
)
def test_defaults_use_only_tar_bz2(monkeypatch: pytest.MonkeyPatch, only_tar_bz2: str):
    """
    Defaults is particular in the sense that it offers both .tar.bz2 and .conda for LOTS
    of packages. SubdirData ignores .tar.bz2 entries if they have a .conda counterpart.
    So if we count all the packages in each implementation, rattler's has way more.
    To remain accurate, we test this with `use_only_tar_bz2`:
        - When true, we only count .tar.bz2
        - When false, we only count .conda
    """
    monkeypatch.setenv("CONDA_USE_ONLY_TAR_BZ2", only_tar_bz2)
    reset_context()
    main_noarch_channel = Channel.from_url("https://repo.anaconda.com/pkgs/main/noarch")
    rattler_index = RattlerIndexHelper.from_platform_aware_channel(main_noarch_channel)
    assert len(rattler_index._index) == 1

    rattler_dot_conda_total = rattler_index.n_packages(
        filter_=lambda pkg: pkg.url.endswith(".conda")
    )
    rattler_tar_bz2_total = rattler_index.n_packages(
        filter_=lambda pkg: pkg.url.endswith(".tar.bz2")
    )

    conda_dot_conda_total = 0
    conda_tar_bz2_total = 0
    for channel_url in main_noarch_channel.urls(subdirs=("noarch",)):
        conda_index = SubdirData(Channel(channel_url))
        conda_index.load()
        for pkg in conda_index.iter_records():
            if pkg["url"].endswith(".conda"):
                conda_dot_conda_total += 1
            elif pkg["url"].endswith(".tar.bz2"):
                conda_tar_bz2_total += 1
            else:
                raise RuntimeError(f"Unrecognized package URL: {pkg['url']}")

    if only_tar_bz2:
        assert conda_tar_bz2_total == rattler_tar_bz2_total
        assert rattler_dot_conda_total == conda_dot_conda_total == 0
    else:
        assert conda_dot_conda_total == rattler_dot_conda_total
        assert conda_tar_bz2_total == rattler_tar_bz2_total


def test_reload_channels(tmp_path: Path):
    (tmp_path / "noarch").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "mamba_repo" / "noarch" / "repodata.json", tmp_path / "noarch")
    initial_repodata = (tmp_path / "noarch" / "repodata.json").read_text()
    index = RattlerIndexHelper(channels=[Channel(str(tmp_path))])
    initial_count = index.n_packages()
    SubdirData._cache_.clear()

    data = json.loads(initial_repodata)
    package = data["packages"]["test-package-0.1-0.tar.bz2"]
    data["packages"]["test-package-copy-0.1-0.tar.bz2"] = {**package, "name": "test-package-copy"}
    modified_repodata = json.dumps(data)
    (tmp_path / "noarch" / "repodata.json").write_text(modified_repodata)

    assert initial_repodata != modified_repodata
    # TODO: Remove this sleep after addressing
    # https://github.com/conda/conda/issues/13783
    time.sleep(1)
    index.reload_channel(Channel(str(tmp_path)))
    assert index.n_packages() == initial_count + 1


@pytest.mark.parametrize(
    "load_type,requested",
    [
        ("shard", ("python",)),
        ("shard", ("django", "celery")),
        ("shard", ("vaex",)),
        ("repodata", ("vaex",)),
        ("main", ()),
    ],
    ids=["shard-small", "shard-medium", "shard-large", "noshard", "main"],
)
def test_load_channel_repo_info_shards(
    load_type: str,
    requested: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise sharded vs classic repodata loading (networked).

    Shard cases must return a non-empty index with fewer packages than the full
    repodata.json for the same channel, confirming the subset path was taken.
    The noshard and main cases use full repodata.json and serve as the baseline.
    """
    load_channel = "defaults" if load_type == "main" else CONDA_FORGE_WITH_SHARDS

    monkeypatch.setattr(context.plugins, "use_sharded_repodata", load_type == "shard")
    assert _is_sharded_repodata_enabled() == (load_type == "shard")

    if load_type == "shard":
        shards_mod = pytest.importorskip(
            "conda.gateways.shards",
            reason="conda.gateways.shards not available; install conda from refactor-sharded2",
        )
        build_repodata_subset = shards_mod.build_repodata_subset
    else:
        build_repodata_subset = None

    in_state = SolverInputState(str(tmp_path / "env"), requested=requested)
    index_helper = RattlerIndexHelper(
        channels=[Channel(f"{load_channel}/{context.subdir}")],
        subdirs=(
            "noarch",
            context.subdir,
        ),
        in_state=in_state,
        build_repodata_subset=build_repodata_subset,
    )

    assert len(index_helper._index) > 0

    if load_type == "shard":
        # Shards deliver a dependency-closure subset — must be smaller than full repodata.
        # Build the full-repodata baseline for the same channel to compare against.
        full_index = RattlerIndexHelper(
            channels=[Channel(f"{load_channel}/{context.subdir}")],
            subdirs=("noarch", context.subdir),
            in_state=in_state,
            build_repodata_subset=None,
        )
        shard_package_count = index_helper.n_packages()
        full_package_count = full_index.n_packages()
        assert shard_package_count > 0, "Shard index must contain at least one package"
        assert shard_package_count < full_package_count, (
            f"Shard index ({shard_package_count} packages) should be a strict subset of "
            f"full repodata ({full_package_count} packages)"
        )
