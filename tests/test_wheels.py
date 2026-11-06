"""The built wheels: fastforks-core carries the permissive model families only, no two wheels share a file.

A plain build of the source tree (`pip install .`) carries every family.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from fastforks._core import registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheels(tmp_path_factory):
    if shutil.which("uv") is None:
        pytest.skip("needs uv to build the wheels")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_dists.py"), "--out", str(out)], check=True)
    return {
        w.name.split("-")[0]: {n for n in zipfile.ZipFile(w).namelist() if ".dist-info/" not in n}
        for w in out.glob("*.whl")
    }


@pytest.fixture(scope="module")
def plain_wheel(tmp_path_factory):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("needs uv to build the wheel")
    out = tmp_path_factory.mktemp("plain")
    subprocess.run([uv, "build", "--quiet", "--wheel", "-o", str(out), str(ROOT)], check=True)
    (wheel,) = out.glob("*.whl")
    return set(zipfile.ZipFile(wheel).namelist())


def _families(files: set[str]) -> set[str]:
    """`task/family` of the model directories in a wheel's file list."""
    return {"/".join(n.split("/")[1:4:2]) for n in files if n.count("/") > 3 and n.split("/")[2] == "models"}


def test_core_wheel_has_exactly_the_permissive_families(wheels):
    want = {f"{c.task}/{c.family}" for t in registry.TASKS for c, _ in registry.cards(t).values()}
    nc = {
        f"{c.task}/{c.family}"
        for t in registry.TASKS
        for c, _ in registry.cards(t).values()
        if c.licence.code_class != "permissive"
    }
    assert nc, "no family outside the core wheel any more: drop fastforks-nc from dists.toml"
    assert _families(wheels["fastforks_core"]) == want - nc
    assert _families(wheels["fastforks_nc"]) == nc
    assert not [n for n in wheels["fastforks_core"] if n.endswith(("golden.toml", ".cu"))]


def test_fastforks_metapackage_has_no_files(wheels):
    assert wheels["fastforks"] == set()


def test_distributions_are_disjoint(wheels):
    assert {"fastforks", "fastforks_core", "fastforks_nc"} <= wheels.keys()
    for (a, fa), (b, fb) in itertools.combinations(wheels.items(), 2):
        assert not fa & fb, f"{a} and {b} both ship {sorted(fa & fb)[:5]}"


def test_plain_build_has_every_family(plain_wheel):
    """TM-1: `pip install .` / `uv pip install .` from a checkout installs every model family, like `-e .`."""
    want = {f"{c.task}/{c.family}" for t in registry.TASKS for c, _ in registry.cards(t).values()}
    assert _families(plain_wheel) == want
    assert not [n for n in plain_wheel if n.endswith(("golden.toml", ".cu"))]
