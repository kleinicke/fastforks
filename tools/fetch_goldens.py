"""Download the golden outputs that the CPU parity tests need into DIR (hash-checked; see the family golden.toml files).

    uv run python tools/fetch_goldens.py --all DIR
    uv run python tools/fetch_goldens.py --changed origin/main DIR

`--changed BASE` selects the families a change since BASE can affect (`affected_families`); `--smallest` keeps
one variant per family (pull requests; the nightly run checks all).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import goldens as store
from fastforks._core import registry
from fastforks._core.files import FileSpec, download, sha256_of

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_URL = "https://huggingface.co/datasets/fastforks/reference-outputs/resolve/main/{file}"  # tools/stage_hf.py


def affected_families(base: str) -> set[str] | None:
    """Families whose CPU outputs a change since `base` can move, or `None` for all of them.

    A family is affected by its own directory and by the `shared/` modules its card `uses`. Kernel libraries are not
    used on CPU (GPU runs check them), docs / tools / other tests cannot move outputs; anything else in `src/` (core,
    task API) and the golden test itself affect every family.
    """
    out = subprocess.run(
        ["git", "-C", ROOT, "diff", "--name-only", f"{base}...HEAD"], check=True, capture_output=True, text=True
    ).stdout
    users = _shared_users()
    families: set[str] = set()
    for name in out.split():
        match Path(name).parts:
            case ("src", "fastforks", _task, "models", family, _, *_):
                families.add(family)
            case ("src", "fastforks", "shared", module, _, *_):
                families |= users.get(module, set())
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
    return families


def _shared_users() -> dict[str, set[str]]:
    """Families by the `shared/` module they use."""
    users: dict[str, set[str]] = {}
    for task in registry.TASKS:
        for card, _ in registry.cards(task).values():
            for m in card.uses:
                users.setdefault(m, set()).add(card.family)
    return users


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
        a.required.write_text("".join(f"{ref}\n" for ref in selected))
    for ref in selected:
        entry = goldens[ref]
        file = f"{ref}.npz"
        dest = a.dir / file
        if dest.is_file() and sha256_of(dest) == entry["sha256"]:
            continue
        download(GOLDEN_URL.format(file=file.replace("@", "%40")), dest, FileSpec(file, entry["sha256"]))
        print(f"{ref}: {dest}")


if __name__ == "__main__":
    main()
