"""Per-agent row fragments: write speed / A/B rows beside the shared files, merge them in one step.

    FASTFORKS_ROWS_AGENT=<agent> python tools/speed_survey.py ...     # rows -> docs/data/speed/_pending/<agent>/
    FASTFORKS_ROWS_AGENT=<agent> python tools/mlx_core_ab.py run ...  # (the same directory, by the target's name)
    python tools/merge_rows.py [--agent A ...] [--dry-run] [--keep]   # fragments -> the shared files, md regenerated

Agents working in parallel all rewrite the same `docs/data/speed/<device>.jsonl` and `docs/plans/mlx-core-ab.jsonl`,
so every rebase conflicts. With `FASTFORKS_ROWS_AGENT` set, `speed_survey` and `mlx_core_ab run` write each row into
`docs/data/speed/_pending/<agent>/<file name of the target>` instead (same replace-by-key merge, so a re-run replaces
the agent's own row). Fragments of different agents never touch the same file.

`merge_rows.py` then merges every fragment into its target: rows replace the target's row of the same key (speed:
ref / mode / dtype / input; A/B: ref / mode / dtype), fragments applied oldest file first so the last writer wins. It
removes the merged fragments (`--keep` leaves them) and rewrites the generated part of `docs/plans/mlx-core-ab.md` when
A/B rows came in. Regenerate the other generated files (`gen_docs.py`, `site_data.py`, ...) after it as usual.

Reads (the `--flagged` selection, a survey's skip of measured rows) still see the shared file only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEED = ROOT / "docs" / "data" / "speed"
PENDING = SPEED / "_pending"
AB = ROOT / "docs" / "plans" / "mlx-core-ab.jsonl"
AGENT_ENV = "FASTFORKS_ROWS_AGENT"
MODES = ("upstream-default", "upstream", "portable", "kernels")


def fragment(target: Path, agent: str | None = None) -> Path:
    """Where a row for `target` goes: the agent's fragment of it when `FASTFORKS_ROWS_AGENT` (or `agent`) is set.

    Only the shared files are redirected (`docs/data/speed/*.jsonl`, `docs/plans/mlx-core-ab.jsonl`): a scratch
    `--out` stays where it is.
    """
    agent = agent if agent is not None else os.environ.get(AGENT_ENV, "")
    t = target.resolve()
    if not agent or (t != AB and t.parent != SPEED):
        return target
    if "/" in agent or agent.startswith("."):
        msg = f"{AGENT_ENV}={agent!r}: a plain name"
        raise ValueError(msg)
    return PENDING / agent / target.name


def target_of(frag: Path) -> Path:
    """The shared file a fragment merges into (by its file name)."""
    return AB if frag.name == AB.name else SPEED / frag.name


def key(path: Path, r: dict[str, Any]) -> tuple:
    """A row's identity in its file: A/B rows by ref / mode / dtype, speed rows also by input."""
    if path.name == AB.name:
        return r["ref"], r["mode"], r["dtype"]
    return r["ref"], r["mode"], r["dtype"], r["input"]


def order(path: Path, r: dict[str, Any]) -> tuple:
    """The sort order of the file (as `mlx_core_ab._merge` / `speed_survey._merge` write it)."""
    if path.name == AB.name:
        return key(path, r)
    mode = MODES.index(r["mode"]) if r["mode"] in MODES else len(MODES)
    return r.get("task", ""), r["ref"], mode, r["dtype"], r["input"]


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a jsonl file (none when it does not exist)."""
    if not path.is_file():
        return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def merge_into(path: Path, new: list[dict[str, Any]], target: Path | None = None) -> int:
    """Replace or add `new` rows in `path` (keys and order of `target`, default `path`); the rows written."""
    target = target or path
    got = {key(target, r): r for r in read(path)}
    got |= {key(target, r): r for r in new}
    rows = sorted(got.values(), key=lambda r: order(target, r))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(new)


def fragments(agents: list[str]) -> list[Path]:
    """The fragment files of `agents` (all when empty), oldest first."""
    if not PENDING.is_dir():
        return []
    dirs = [PENDING / a for a in agents] if agents else sorted(p for p in PENDING.iterdir() if p.is_dir())
    files = [f for d in dirs if d.is_dir() for f in d.glob("*.jsonl")]
    return sorted(files, key=lambda f: (f.stat().st_mtime, str(f)))


def main(argv: list[str] | None = None) -> int:
    """Command line (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", nargs="*", default=[], help="only these agents' fragments")
    ap.add_argument("--dry-run", action="store_true", help="list what would merge")
    ap.add_argument("--keep", action="store_true", help="leave the fragments in place")
    a = ap.parse_args(argv)
    files = fragments(a.agent)
    ab = False
    for f in files:
        rows, target = read(f), target_of(f)
        print(f"{f.relative_to(ROOT)}: {len(rows)} rows -> {target.relative_to(ROOT)}")
        if a.dry_run:
            continue
        merge_into(target, rows)
        ab |= target == AB
        if not a.keep:
            f.unlink()
            if not any(f.parent.iterdir()):
                f.parent.rmdir()
    if not files:
        print("no fragments")
    if ab:
        cmd = [sys.executable, str(Path(__file__).with_name("mlx_core_ab.py")), "md"]
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
