"""Cross-family float16 stress scan: the default model of each family, portable and kernels, on inputs that stress fp16.

    python tools/fp16_stress.py [--task T ...] [--only SUBSTR ...] [--mode portable kernels] [--part I/N]
    [--out F.jsonl]

The task's `bench_inputs` call is made with every 3-channel image tensor replaced by: `noise` (uniform), `noise10`
(low-contrast noise, 0.5 +- 0.05), `const0`, `const1`, `half` (constant 0.5) and
`binary` (saturated 0 / 1). One JSON line per (model, mode, input); `status` is `fail` when a result holds NaN / inf
(or the call raises). Exit code = failures.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import family_smoke as fs

TASKS = ("restoration", "depth", "segmentation", "matting", "edges")
INPUTS = ("noise", "noise10", "const0", "const1", "half", "binary")


def _stress(name: str, like: Any) -> Any:  # noqa: ANN401
    import torch  # noqa: PLC0415

    gen = torch.Generator().manual_seed(1)
    n = torch.rand(like.shape, generator=gen)
    x = {
        "noise": n,
        "noise10": 0.5 + 0.1 * (n - 0.5),
        "const0": torch.zeros_like(n),
        "const1": torch.ones_like(n),
        "half": torch.full_like(n, 0.5),
        "binary": (n > _HALF).float(),
    }[name]
    return x.to(like.device, like.dtype)


_HALF, _NDIM, _RGB = 0.5, 4, 3


def _swap(obj: Any, name: str) -> Any:  # noqa: ANN401
    import torch  # noqa: PLC0415

    if isinstance(obj, torch.Tensor) and obj.ndim == _NDIM and obj.shape[1] == _RGB and obj.dtype.is_floating_point:
        return _stress(name, obj)
    if isinstance(obj, list | tuple):
        return type(obj)(_swap(o, name) for o in obj)
    return obj


def run_one(model_id: str, mode: str) -> list[dict[str, Any]]:
    import torch  # noqa: PLC0415

    import fastforks  # noqa: PLC0415
    from fastforks._core.bench import bench_inputs  # noqa: PLC0415

    base = {"model": model_id, "mode": mode}
    try:
        m = fastforks.load(model_id, mode=mode)
    except Exception as e:  # noqa: BLE001
        kind = type(e).__name__
        return [
            {
                **base,
                "input": "-",
                "status": "skip" if "Unavailable" in kind or "Licence" in kind or "Missing" in kind else "fail",
                "error": f"load: {kind}: {str(e)[:200]}",
            }
        ]
    rows = []
    try:
        args, kw = bench_inputs(m, fs._size(m))  # noqa: SLF001
        for name in INPUTS:
            row = {**base, "input": name, "status": "ok"}
            try:
                res = m(*_swap(args, name), **kw)
                torch.cuda.synchronize()
                bad = fs.check(res)
                if bad:
                    row["status"], row["error"] = "fail", "; ".join(bad)
            except Exception as e:  # noqa: BLE001
                row["status"], row["error"] = "fail", f"call: {type(e).__name__}: {str(e)[:200]}"
            rows.append(row)
    finally:
        del m
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", nargs="+", default=list(TASKS))
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--mode", nargs="+", default=["portable", "kernels"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out", default="fp16_stress.jsonl")
    a = ap.parse_args()
    fails = 0
    with Path(a.out).open("a") as f:
        for task, mid in fs.default_models(a.only, a.task, a.part):
            for mode in fs.modes_of(mid, task, tuple(a.mode)):
                for row in run_one(mid, mode):
                    row["task"] = task
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    if row["status"] == "fail":
                        fails += 1
                        print("FAIL", row, flush=True)
                print("done", task, mid, mode, flush=True)
    return fails


if __name__ == "__main__":
    sys.exit(main())
