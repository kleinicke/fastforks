"""Outputs with and without the load-time arena compaction (`shared.arena.compact_runner`), per (ref, mode).

    python tools/arena_compare.py --out rows.jsonl [--mode portable kernels] REF [REF ...]

Per (ref, mode): load with compaction (default), run the documented bench inputs (same seed), load again with
`compact_runner` disabled, run again; compare every output array bit for bit. A difference is checked against a
second compacted run (non-deterministic kernels differ from run to run too). One JSON line per pair:
`{ref, mode, status: same | noise | DIFF | error, max_abs, alloc_mb_on/off, reserved_mb_on/off}`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _outputs(model_id: str, mode: str, task: str | None, compact: bool) -> tuple[dict[str, Any], float, float]:
    import torch  # noqa: PLC0415

    import fastforks  # noqa: PLC0415
    from fastforks._core import model as core_model  # noqa: PLC0415
    from fastforks._core.bench import bench_inputs  # noqa: PLC0415
    from family_smoke import _size  # noqa: PLC0415

    real = core_model.compact_runner
    core_model.compact_runner = real if compact else (lambda _r: None)
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = fastforks.load(model_id, mode=mode)
        alloc = torch.cuda.memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
        torch.manual_seed(0)
        args, kw = bench_inputs(m, _size(m))
        res = m(*args, **kw)
        r = res[0] if isinstance(res, list | tuple) else res
        out = {k: (v if isinstance(v, tuple) else (v,)) for k, v in r.numpy().items()}
        del m, res, r
        return out, alloc, reserved
    finally:
        core_model.compact_runner = real
        gc.collect()
        torch.cuda.empty_cache()


def _diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    worst = 0.0
    if a.keys() != b.keys():
        return float("inf")
    for k in a:
        for x, y in zip(a[k], b[k], strict=True):
            if x is None and y is None:
                continue
            if x.shape != y.shape:
                return float("inf")
            if not np.array_equal(x, y, equal_nan=True):
                worst = max(worst, float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.dtype.kind in "fiu" else 1.0)
    return worst


def main() -> None:
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--mode", nargs="+", default=["portable", "kernels"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("FASTFORKS_NONCOMMERCIAL", "1")
    from fastforks._core import registry  # noqa: PLC0415

    with Path(a.out).open("a") as f:
        for ref in a.refs:
            for mode in a.mode:
                row: dict[str, Any] = {"ref": ref, "mode": mode}
                try:
                    task = next(t for t in registry.TASKS if ref in registry.cards(t))
                    if mode != "auto" and mode not in registry.cards(task)[ref][0].modes:
                        continue
                    row["task"] = task
                    on, row["alloc_mb_on"], row["reserved_mb_on"] = _outputs(ref, mode, task, True)
                    off, row["alloc_mb_off"], row["reserved_mb_off"] = _outputs(ref, mode, task, False)
                    d = _diff(on, off)
                    row["max_abs"] = d
                    if d == 0:
                        row["status"] = "same"
                    else:
                        on2, *_ = _outputs(ref, mode, task, True)
                        d2 = _diff(on, on2)
                        row["self_noise"] = d2
                        row["status"] = "noise" if d <= d2 else "DIFF"
                except Exception as e:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["status"], ref, mode, row.get("max_abs", ""), row.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
