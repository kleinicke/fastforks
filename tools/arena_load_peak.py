"""Peak reserved memory DURING load and load wall time, with and without the arena compaction.

    python tools/arena_load_peak.py --out rows.jsonl [--mode portable kernels] [--rounds 3] REF [REF ...]

Per (ref, mode): alternate compaction on / off `rounds` times (model deleted and `empty_cache` between loads); per load
`peak_reserved_mb` (max_memory_reserved since the load started), `after_reserved_mb`, `alloc_mb`, `load_s`. One JSON line
per (ref, mode, compact) with the median over the rounds, or an `error` row (e.g. an OOM). Run it as a GPU job.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path


def _load(model_id: str, mode: str, compact: bool) -> dict[str, float]:
    import torch  # noqa: PLC0415

    import fastforks  # noqa: PLC0415

    os.environ["FASTFORKS_ARENA"] = "1" if compact else "0"
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        m = fastforks.load(model_id, mode=mode)
        torch.cuda.synchronize()
        row = {
            "load_s": time.perf_counter() - t,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "after_reserved_mb": torch.cuda.memory_reserved() / 2**20,
            "alloc_mb": torch.cuda.memory_allocated() / 2**20,
        }
        del m
        return row
    finally:
        os.environ.pop("FASTFORKS_ARENA", None)
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--mode", nargs="+", default=["portable", "kernels"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("FASTFORKS_NONCOMMERCIAL", "1")
    with Path(a.out).open("a") as f:
        for ref in a.refs:
            for mode in a.mode:
                res: dict[bool, list[dict[str, float]]] = {True: [], False: []}
                err = None
                try:
                    _load(ref, mode, compact=False)  # warm: file cache, kernel build / JIT
                    for _ in range(a.rounds):
                        for c in (True, False):
                            res[c].append(_load(ref, mode, c))
                except Exception as e:  # noqa: BLE001 - recorded
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                for c, rows in res.items():
                    row: dict = {"ref": ref, "mode": mode, "compact": c, "rounds": len(rows)}
                    if rows:
                        row |= {k: round(statistics.median(r[k] for r in rows), 3) for k in rows[0]}
                    if err:
                        row["error"] = err
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(row, flush=True)


if __name__ == "__main__":
    main()
