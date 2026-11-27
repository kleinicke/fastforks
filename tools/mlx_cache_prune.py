"""Dry-run-first pruning of the MLX conversion cache: what `fastforks cache mlx --prune` would delete, by reason.

    python tools/mlx_cache_prune.py [--root DIR] [--family NAME] [--apply]

Keeps, per (variant, revision, weight file, dtype, layout name), the newest file (`mlx_cache.stale_reason`: files of
older port code, development files of `FASTFORKS_MLX_CACHE_SRC=0`, other revisions or weight files, variants no longer
installed, files without provenance). Lists them with the space they hold and deletes nothing unless `--apply` is
given. The shared cache of the Mac's agents is never pruned with `--apply` by an agent: run the dry run and report.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from fastforks._core import cli, mlx_cache


def report(removed: list[tuple[mlx_cache.Entry, str]], root: Path, *, applied: bool) -> list[str]:
    """The lines printed: one per file, then the totals per kind of reason."""
    lines = []
    kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    verb = "removed" if applied else "would remove"
    for e, reason in removed:
        lines.append(f"{verb}  {e.path.relative_to(root)}  {e.size / 2**20:,.0f} MB  ({reason})")
        kind = reason.split(" ", 1)[0] if reason.startswith(("superseded", "revision")) else reason
        kinds[kind][0] += 1
        kinds[kind][1] += e.size
    for kind, (n, size) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"{n:5d} files {size / 2**30:8.2f} GB  {kind}")
    total = sum(e.size for e, _ in removed)
    lines.append(
        f"{len(removed)} stale files, {total / 2**30:.2f} GB {'freed' if applied else 'reclaimable (dry run)'}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Command line (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="the MLX cache directory (default <FASTFORKS_CACHE>/mlx)")
    ap.add_argument("--family", help="only files under <task>/<family>/ of this family")
    ap.add_argument("--apply", action="store_true", help="delete the stale files (default: list them only)")
    a = ap.parse_args(argv)
    root = (a.root or mlx_cache.cache_dir()).absolute()
    removed = mlx_cache.prune(cli._all_cards(), dry_run=True, root=root)  # noqa: SLF001 - the CLI's card index
    if a.family:
        removed = [(e, r) for e, r in removed if e.path.parent.name == a.family]
    if a.apply:
        for e, _ in removed:
            if e.path.is_symlink() or not e.path.resolve().is_relative_to(root.resolve()):
                print(f"refusing to delete {e.path}: not a file inside {root}", file=sys.stderr)
                return 1
            e.path.unlink()
    for line in report(removed, root, applied=a.apply):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
