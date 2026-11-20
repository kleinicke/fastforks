"""Real-use smoke of every family in every mode: the default model, one forward at a small size, finite typed output.

    python tools/family_smoke.py [--mode auto kernels portable] [--only SUBSTR ...] [--task T ...] [--part I/N]
        [--out results.jsonl]

For each family (the card's first variant) and each mode the card lists (`auto`, `kernels`, `portable`) it loads the
model through the public `fastforks.load`, calls it once on the task's `bench_inputs` (the documented minimal input,
image side ~256 px) and checks the result: a fastforks result type, at least one tensor, every tensor finite.
One JSON line per (model, mode) goes to `--out`; the exit code is the number of failures. `--part I/N` takes every
N-th model (chunks for the locked GPU commands). Used by `tests/test_family_smoke.py` (one parametrized slow GPU
test); run for all families by hand before a release.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

MODES = ("auto", "kernels", "portable")


def default_models(
    only: list[str] | None = None, tasks: list[str] | None = None, part: str = "0/1"
) -> list[tuple[str, str]]:
    """`[(task, model id)]`: the first variant of each family, sorted by task then id."""
    from fastforks._core import registry  # noqa: PLC0415

    found: dict[tuple[str, str], str] = {}
    for task in registry.TASKS:
        for mid, (card, _v) in registry.cards(task).items():
            found.setdefault((task, card.family), mid)
    out = sorted((t, m) for (t, _f), m in found.items())
    out = [(t, m) for t, m in out if (not only or any(s in m for s in only)) and (not tasks or t in tasks)]
    i, n = map(int, part.split("/"))
    return out[i::n]


def _size(model: Any) -> Any:  # noqa: ANN401
    from fastforks._core.bench import BenchSize  # noqa: PLC0415
    from fastforks._core.sizes import size_range  # noqa: PLC0415

    sr = size_range(model.variant, model.card)
    mult = max(sr.multiple, 1)
    side = max(256, sr.reject_below, sr.min_size)
    side = -(-side // mult) * mult
    if model.card.task == "volume":
        return BenchSize(32, 64, 4)
    if model.card.task in ("multiview", "tracking", "interpolation"):
        return BenchSize(side, side + side // 4, 4)
    if model.card.task == "restoration":  # face restorers take exactly one square crop
        return BenchSize(side, side, 8)
    return BenchSize(side, side + side // 4, 8)


def check(result: Any) -> list[str]:  # noqa: ANN401
    """Problems of one result (empty: fine): a typed result with at least one tensor, all finite."""
    problems: list[str] = []
    r = result[0] if isinstance(result, list | tuple) and result and hasattr(result[0], "numpy") else result
    if not hasattr(r, "numpy") or not hasattr(r, "model"):
        return [f"not a fastforks result: {type(result).__name__}"]
    arrays = r.numpy()
    if not arrays:
        problems.append("result holds no tensor")
    for name, a in arrays.items():
        parts = a if isinstance(a, tuple) else (a,)
        problems += [
            f"{name}: NaN / inf" for p in parts if p is not None and p.dtype.kind == "f" and not np.isfinite(p).all()
        ]
    return problems


def run_one(model_id: str, mode: str, device: str | None = None) -> dict[str, Any]:
    """Load `model_id` in `mode`, one forward on the bench inputs; `{model, mode, status, ...}`."""
    import torch  # noqa: PLC0415

    import fastforks  # noqa: PLC0415
    from fastforks._core.bench import bench_inputs  # noqa: PLC0415

    row: dict[str, Any] = {"model": model_id, "mode": mode, "status": "ok"}
    t0 = time.perf_counter()
    try:
        m = fastforks.load(model_id, mode=mode, **({"device": device} if device else {}))
    except Exception as e:  # noqa: BLE001
        kind = type(e).__name__
        row["status"] = (
            "skip" if kind in {"BackendUnavailableError", "MissingDependencyError", "LicenceError"} else "fail"
        )
        row["error"] = f"load: {kind}: {str(e)[:300]}"
        return row
    try:
        args, kw = bench_inputs(m, _size(m))
        row["load_s"] = round(time.perf_counter() - t0, 1)
        t1 = time.perf_counter()
        res = m(*args, **kw)
        torch.cuda.synchronize()
        row["call_s"] = round(time.perf_counter() - t1, 1)
        row["backend"] = str(getattr(res[0] if isinstance(res, list | tuple) else res, "backend", ""))
        bad = check(res)
        if bad:
            row["status"], row["error"] = "fail", "; ".join(bad)
    except Exception as e:  # noqa: BLE001
        row["status"] = "fail"
        row["error"] = f"call: {type(e).__name__}: {str(e)[:300]}"
        row["trace"] = traceback.format_exc()[-1500:]
    finally:
        del m
        gc.collect()
        torch.cuda.empty_cache()
    return row


def modes_of(model_id: str, task: str, wanted: tuple[str, ...]) -> list[str]:
    from fastforks._core import registry  # noqa: PLC0415

    card = registry.cards(task)[model_id][0]
    return [m for m in wanted if m == "auto" or m in card.modes]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--task", nargs="+")
    ap.add_argument("--part", default="0/1", help="I/N: every N-th model starting at I (chunks)")
    ap.add_argument("--out", default="family_smoke.jsonl")
    ap.add_argument("--device")
    a = ap.parse_args()
    os.environ.setdefault("FASTFORKS_NONCOMMERCIAL", "1")
    fails = 0
    with Path(a.out).open("a", encoding="utf-8") as f:
        for task, mid in default_models(a.only, a.task, a.part):
            for mode in modes_of(mid, task, tuple(a.mode)):
                row = run_one(mid, mode, a.device)
                row["task"] = task
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(row["status"].upper(), task, mid, mode, row.get("call_s", ""), row.get("error", ""), flush=True)
                fails += row["status"] == "fail"
    return fails


if __name__ == "__main__":
    sys.exit(main())
