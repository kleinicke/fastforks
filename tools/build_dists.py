"""Build the core wheel, the model distributions of dists.toml and the `fastforks` metapackage into dist/.

    uv run python tools/build_dists.py [DIST ...]

The core wheel `fastforks-core` ships every model family whose code licence is permissive (`CORE_CLASSES`);
dists.toml cuts the other families into separate distributions by class of the code licence, across all tasks. Each is
staged in a temporary directory: the selected model directories, their upstream licence files (listed in the wheel
metadata), and a generated pyproject.toml pinning the same fastforks-core version and the libraries its models need
(the cards' `ModelCard.requires`, with the version specs of the core pyproject extras). The core wheel gets one extra
per distribution (`fastforks-core[nc]` = fastforks-nc of the same version); its licence expression and licence files
gain those of the families it ships. `fastforks` is a metapackage without code: fastforks-core with every distribution
and the `libs` extra (every optional library but TensorRT), so `pip install fastforks` installs everything. All of this
is added at build time only: the distributions are not on an index that `uv lock` could resolve. A plain `uv build`
(or `pip install .`) of the source tree is one wheel with every model family, like `pip install -e .`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from fastforks._core import registry
from fastforks._core.cards import LIBRARY_EXTRA, ModelCard

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


CORE = "fastforks-core"
META = "fastforks"
CORE_CLASSES = ("permissive",)
"""Code-licence classes whose model families ship in the core wheel."""


def _cards(spec: dict) -> list[ModelCard]:
    """One card per model family of a dists.toml entry: its tasks (default: all) and classes of the code licence."""
    tasks, classes = spec.get("tasks", registry.TASKS), spec["code_classes"]
    found = {(card.task, card.family): card for t in tasks for card, _ in registry.cards(t).values()}
    return [card for _, card in sorted(found.items()) if card.licence.code_class in classes]


def select(spec: dict) -> list[Path]:
    """Model directories of a dists.toml entry (sorted by task, then family)."""
    return [ROOT / "src" / "fastforks" / card.task / "models" / card.family for card in _cards(spec)]


def library_requirements(spec: dict, extras: dict[str, list[str]]) -> list[str]:
    """Requirement strings of the optional libraries the models of a dists.toml entry need.

    Args:
        spec: The dists.toml entry.
        extras: The core pyproject `[project.optional-dependencies]`.

    Returns:
        The requirements (version specs from the extras that install the needed libraries), sorted, no duplicates.
    """
    wanted = {LIBRARY_EXTRA[lib] for card in _cards(spec) for lib in card.requires}
    return sorted({req for extra in wanted for req in extras[extra]})


def _licences(dirs: list[Path]) -> tuple[set[str], list[str]]:
    """The code licence expressions and the licence files (relative to the source root) of model directories."""
    licences, files = set(), []
    for d in dirs:
        card = tomllib.loads((d / "card.toml").read_text())
        licences.add(card["licence"]["code"])
        files += [str((d / f).relative_to(ROOT)) for f in card["licence"]["files"]]
    return licences, files


def _expression(licences: set[str]) -> str:
    """`a AND b AND ...` over SPDX expressions; compound ones in parentheses unless they are plain AND chains."""
    terms = set()
    for x in licences:
        terms |= set(x.split(" AND ")) if " OR " not in x and "(" not in x else {f"({x})"}
    return " AND ".join(sorted(terms))


def stage(name: str, spec: dict, version: str, out: Path, extras: dict[str, list[str]] | None = None) -> None:
    dirs = select(spec)
    for d in dirs:
        shutil.copytree(
            d, out / d.relative_to(ROOT), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "golden.toml")
        )
    licences, files = _licences(dirs)
    include = [str(d.relative_to(ROOT)) for d in dirs]
    if extras is None:
        extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    deps = [f"{CORE}=={version}", *library_requirements(spec, extras)]
    pyproject = f"""\
[project]
name = "{name}"
version = "{version}"
description = "{spec["summary"]}"
requires-python = ">=3.11"
license = "{_expression(licences)}"
license-files = {files!r}
dependencies = {json.dumps(deps)}

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
only-include = {json.dumps(include)}
sources = ["src"]
"""
    (out / "pyproject.toml").write_text(pyproject)
    (out / "README.md").write_text(f"# {name}\n\n{spec['summary']}\n")


_MODELS_EXCLUDE = '"src/fastforks/*/models/*/golden.toml"'


def core_pyproject(text: str, dists: dict, version: str) -> str:
    """Return the core pyproject.toml: the permissive model families, their licences and one extra per distribution.

    Args:
        text: The source pyproject.toml.
        dists: The parsed dists.toml.
        version: The fastforks version.

    Returns:
        The pyproject.toml text for the core wheel.
    """
    if META in dists or CORE in dists:
        msg = f"dists.toml: {META} and {CORE} are built from pyproject.toml, not from dists.toml"
        raise ValueError(msg)
    project = tomllib.loads(text)["project"]
    licences, files = _licences(select({"code_classes": CORE_CLASSES}))
    lines = text.splitlines(keepends=True)
    for key, value in (
        ("license", json.dumps(_expression({project["license"], *licences}))),
        ("license-files", json.dumps([*project["license-files"], *files])),
    ):
        i = next((i for i, x in enumerate(lines) if x.startswith(f"{key} = ")), None)
        if i is None:
            msg = f"pyproject.toml has no top-level `{key} = ...` line"
            raise ValueError(msg)
        lines[i] = f"{key} = {value}\n"
    text = "".join(lines)
    if _MODELS_EXCLUDE not in text:
        msg = f"pyproject.toml: the wheel exclude has no {_MODELS_EXCLUDE}"
        raise ValueError(msg)
    others = [str(d.relative_to(ROOT)) + "/" for spec in dists.values() for d in select(spec)]
    exclude = ", ".join([*(json.dumps(x) for x in others), _MODELS_EXCLUDE])
    text = text.replace(_MODELS_EXCLUDE, exclude, 1)
    extras = "".join(f'{name.removeprefix("fastforks-")} = ["{name}=={version}"]\n' for name in dists)
    head = "[project.optional-dependencies]\n"
    if head not in text:
        msg = "pyproject.toml has no [project.optional-dependencies] table"
        raise ValueError(msg)
    return text.replace(head, head + extras, 1)


def build_core(dists: dict, version: str, out: Path = DIST) -> None:
    """Build the core wheel from a staged copy of the tree with the pyproject.toml of `core_pyproject`."""
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp)
        for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(ROOT / name, stage_dir / name)
        shutil.copytree(ROOT / "src", stage_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.so"))
        text = core_pyproject((ROOT / "pyproject.toml").read_text(), dists, version)
        (stage_dir / "pyproject.toml").write_text(text)
        subprocess.run(["uv", "build", "--quiet", "--wheel", "-o", out, stage_dir], check=True)


def meta_pyproject(text: str, dists: dict, version: str) -> str:
    """Return the pyproject.toml of the `fastforks` metapackage: fastforks-core with every distribution and `libs`.

    Args:
        text: The source pyproject.toml (for the description, authors, classifiers and the `libs` extra check).
        dists: The parsed dists.toml.
        version: The fastforks-core version.

    Returns:
        The pyproject.toml text of a wheel without code that pins fastforks-core of the same version.
    """
    project = tomllib.loads(text)["project"]
    if "libs" not in project["optional-dependencies"]:
        msg = "pyproject.toml has no `libs` extra (every optional library), which the metapackage installs"
        raise ValueError(msg)
    picks = ",".join([*(name.removeprefix("fastforks-") for name in dists), "libs"])
    tables = (", ".join(f"{k} = {json.dumps(v)}" for k, v in a.items()) for a in project["authors"])
    authors = ", ".join("{" + x + "}" for x in tables)  # TOML inline tables
    return f"""\
[project]
name = "{META}"
version = "{version}"
description = {json.dumps(project["description"])}
readme = "README.md"
license = "Apache-2.0"
license-files = ["LICENSE"]
authors = [{authors}]
requires-python = {json.dumps(project["requires-python"])}
classifiers = {json.dumps(project["classifiers"])}
dependencies = ["{CORE}[{picks}]=={version}"]

[project.optional-dependencies]
tensorrt = ["{CORE}[tensorrt]=={version}"]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
bypass-selection = true
"""


def build_meta(dists: dict, version: str, out: Path = DIST) -> None:
    """Build the `fastforks` metapackage (no code; README and LICENSE of the project)."""
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp)
        for name in ("README.md", "LICENSE"):
            shutil.copy2(ROOT / name, stage_dir / name)
        (stage_dir / "pyproject.toml").write_text(meta_pyproject((ROOT / "pyproject.toml").read_text(), dists, version))
        subprocess.run(["uv", "build", "--quiet", "--wheel", "-o", out, stage_dir], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dists", nargs="*", help=f"names from dists.toml, {CORE} or {META} (default: all)")
    ap.add_argument("--out", type=Path, default=DIST, help="output directory (default: dist/)")
    a = ap.parse_args()
    dists = tomllib.loads((ROOT / "dists.toml").read_text())
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    for name in a.dists or [CORE, META, *dists]:
        if name in (CORE, META):
            (build_core if name == CORE else build_meta)(dists, version, a.out)
            continue
        if not select(dists[name]):
            print(f"{name}: no models yet, skipped")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            stage(name, dists[name], version, Path(tmp))
            subprocess.run(["uv", "build", "--quiet", "--wheel", "-o", a.out, tmp], check=True)


if __name__ == "__main__":
    main()
