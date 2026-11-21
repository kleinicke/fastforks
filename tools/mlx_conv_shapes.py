"""Record the convolution shapes an MLX graph runs: one forward of a survey job, every `aten.convolution` call.

Wraps `shared.mlx_graph._conv_core` while the model's first call traces its graph (`FF_MLX_NO_TUNE=1`, so a tuned
segment is not traced twice) and writes one row per distinct (input, weight, stride, padding, dilation, transposed,
groups, dtype) with its count per forward. `tools/mlx_conv_bench.py` times the rows.

    FASTFORKS_OFFLINE=1 PYTHONPATH=src $PY tools/mlx_conv_shapes.py rife-4.25 --dtype float16 --out rife.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("FF_MLX_NO_TUNE", "1")
sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    """Run one survey job on MLX and dump its conv shapes."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--mode", default="portable")
    ap.add_argument("--input", help="HxW (default: the survey's input)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch  # noqa: PLC0415

    import speed_survey  # noqa: PLC0415
    from fastforks.shared import mlx_graph  # noqa: PLC0415

    dev = speed_survey._device("apple-m1-pro-mlx-16gb")
    job = next(j for j in speed_survey.plan(dev, only=[a.ref], modes=[a.mode]) if j["dtype"] == a.dtype)
    if a.input:
        job["input"] = a.input
    seen: Counter[tuple] = Counter()
    orig = mlx_graph._conv_core

    def wrap(x, w, b, stride, padding, dilation, transposed, output_padding, groups, act=0, slope=None):  # noqa: ANN001, ANN202, PLR0913
        seen[
            (
                tuple(x.shape), tuple(w.shape), tuple(stride), tuple(padding), tuple(dilation), bool(transposed),
                int(groups), tuple(output_padding), str(x.dtype).removeprefix("mlx.core."), b is not None,
            )
        ] += 1  # fmt: skip
        return orig(x, w, b, stride, padding, dilation, transposed, output_padding, groups, act, slope)

    mlx_graph._conv_core = wrap
    task = __import__(f"fastforks.{job['task']}", fromlist=["load"])
    model = task.load(a.ref, mode=a.mode, device="mlx", dtype=a.dtype)
    args, kw, _ = speed_survey._inputs(model, job)
    with torch.inference_mode():
        model(*args, **kw)
    keys = ("x", "w", "stride", "padding", "dilation", "transposed", "groups", "output_padding", "dtype", "bias")
    rows = [{**dict(zip(keys, k, strict=True)), "count": n} for k, n in seen.items()]
    Path(a.out).write_text(json.dumps({"ref": a.ref, "input": job["input"], "convs": rows}, indent=1))
    print(f"{a.ref}: {len(rows)} shapes, {sum(seen.values())} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
