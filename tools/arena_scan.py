"""Allocator fragmentation after load: reserved - allocated (MB) of every (ref, mode) of the speed floor.

    python tools/arena_scan.py --out rows.jsonl [--task TASK ...] [--only REF ...]

One process loads the refs one after the other (model deleted and `empty_cache` between them), per ref and mode
(portable, kernels) it records `alloc_mb`, `reserved_mb` and `frag_mb` right after `load` (nothing run). Failures
become `error` rows. Run it as a GPU job in a capped unit.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ROOT / "docs" / "data" / "speed.jsonl"


def jobs(tasks: set[str], only: set[str]) -> list[tuple[str, str, str]]:
    seen: dict[tuple[str, str, str], None] = {}
    for line in FLOOR.read_text().splitlines():
        r = json.loads(line)
        if r["mode"] not in {"portable", "kernels"}:
            continue
        if (tasks and r["task"] not in tasks) or (only and r["ref"] not in only):
            continue
        seen[(r["task"], r["ref"], r["mode"])] = None
    return sorted(seen)


def main() -> None:
    import torch  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    done = set()
    out = Path(a.out)
    if out.exists():
        done = {(r["ref"], r["mode"]) for r in map(json.loads, out.read_text().splitlines())}
    for task, ref, mode in jobs(set(a.task), set(a.only)):
        if (ref, mode) in done:
            continue
        row: dict = {"task": task, "ref": ref, "mode": mode}
        try:
            gc.collect()
            torch.cuda.empty_cache()
            m = importlib.import_module(f"fastforks.{task}").load(ref, mode=mode, device="cuda")
            torch.cuda.synchronize()
            al, rs = torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20
            row |= {"alloc_mb": round(al, 1), "reserved_mb": round(rs, 1), "frag_mb": round(rs - al, 1)}
            del m
        except Exception as e:  # noqa: BLE001 - recorded
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
