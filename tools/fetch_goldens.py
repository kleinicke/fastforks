"""Download the golden outputs that the CPU parity tests need into DIR (hash-checked; see the family golden.toml files).

    uv run python tools/fetch_goldens.py --all DIR
    uv run python tools/fetch_goldens.py --changed origin/main DIR

`--changed BASE` selects the families a change since BASE can affect (`affected_families`); `--smallest` keeps
one variant per family (pull requests; the nightly run checks all).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

import goldens as store
from fastforks._core import registry
from fastforks._core.files import FileSpec, download, sha256_of
from select_tests import git_changed, select

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_URL = "https://huggingface.co/datasets/fastforks/reference-outputs/resolve/main/{file}"  # tools/stage_hf.py


def affected_families(base: str) -> set[str] | None:
    """Families whose CPU outputs a change since `base` can move, or `None` for all of them.

    Model and shared changes use the release selector's import closure. Kernel libraries are checked on GPU.
    Core, task API and golden harness changes affect every family.
    """
    paths = git_changed(base, "HEAD", ROOT, from_merge_base=True)
    families: set[str] = set()
    model_paths = []
    for name in paths:
        match Path(name).parts:
            case ("src", "fastforks", _task, "models", family, _, *_):
                families.add(family)
                model_paths.append(name)
            case ("src", "fastforks", "shared", _, *_):
                model_paths.append(name)
            case ("src", "fastforks", "kernels", *_):
                pass
            case ("tests", "test_golden.py" | "synth.py" | "conftest.py"):
                return None
            case ("tests", *_):
                pass
            case ("src", *_):
                return None
            case _:
                pass  # docs, tools, CI
    if model_paths:
        families.update(select(model_paths, ROOT).families)
    return families


def smallest(refs: list[str]) -> list[str]:
    """The variant with the smallest weights of each family in `refs`."""
    best: dict[str, tuple[int, str]] = {}
    for task in registry.TASKS:
        for card, v in registry.cards(task).values():
            if v.ref in refs:
                size = (sum(w.size or 0 for w in v.weights), v.ref)
                best[card.family] = min(best.get(card.family, size), size)
    return sorted(ref for _, ref in best.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true")
    which.add_argument("--changed", metavar="BASE")
    ap.add_argument("dir", type=Path)
    ap.add_argument("--required", type=Path, help="also write the selected refs to this file, one per line")
    ap.add_argument("--smallest", action="store_true", help="only the smallest variant of each selected family (PRs)")
    a = ap.parse_args()
    goldens = {ref: e for fam in store.load_all(registry.TASKS).values() for ref, e in fam["refs"].items()}
    families = None if a.all else affected_families(a.changed)
    owner = {v.ref: card.family for task in registry.TASKS for card, v in registry.cards(task).values()}
    selected = [ref for ref in sorted(goldens) if families is None or owner[ref] in families]
    if a.smallest:
        selected = smallest(selected)
    if a.required:
        a.required.write_text("".join(f"{ref}\n" for ref in selected), encoding="utf-8")
    for ref in selected:
        entry = goldens[ref]
        file = f"{ref}.npz"
        dest = a.dir / file
        if dest.is_file() and sha256_of(dest) == entry["sha256"]:
            continue
        remote = quote(entry.get("file", file), safe="")
        download(GOLDEN_URL.format(file=remote), dest, FileSpec(file, entry["sha256"]))
        print(f"{ref}: {dest}")


if __name__ == "__main__":
    main()
