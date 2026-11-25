"""Time the conv shapes of `tools/mlx_conv_shapes.py` on MLX (the graph's lowering, portable and kernels) and torch MPS.

Per distinct shape: the median ms of `mlx_graph._conv_core` in portable mode (`port`), in kernels mode (`kern`), and of
`torch.nn.functional.conv*` on MPS in NCHW (`mps`, how the MPS path runs it); each call is repeated `--reps` times per
evaluation so launch cost is amortised as in a graph. The totals weight each shape by its count per forward.

    PYTHONPATH=src $PY tools/mlx_conv_bench.py rife.json [more.json ...] [--dtype float16] [--min-ms 0.2]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _median(f, rounds: int = 7) -> float:  # noqa: ANN001
    f()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts)


def bench(row: dict, dtype: str, reps: int, *, mps: bool = True) -> dict[str, float]:
    """Ms per call of one recorded conv: port, kern and (with `mps`) mps."""
    import mlx.core as mx  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from fastforks.shared import mlx_graph as g  # noqa: PLC0415

    mdt = getattr(mx, dtype)
    rng = np.random.default_rng(0)
    xs, ws = row["x"], row["w"]
    x = rng.standard_normal(xs).astype(np.float32)
    w = (rng.standard_normal(ws) / np.sqrt(np.prod(ws[1:]))).astype(np.float32)
    bvec = rng.standard_normal(ws[1] if row["transposed"] else ws[0]).astype(np.float32) if row["bias"] else None
    perm = (0, *range(2, len(xs)), 1)
    xl = mx.array(np.ascontiguousarray(np.transpose(x, perm))).astype(mdt)
    xm = mx.transpose(xl, (0, len(xs) - 1, *range(1, len(xs) - 1)))  # NCHW view of channels-last data, as in a graph
    wm = mx.array(w).astype(mdt)
    bm = mx.array(bvec).astype(mdt) if bvec is not None else None
    mx.eval(xl, wm, bm) if bm is not None else mx.eval(xl, wm)
    out = {}
    op = row.get("output_padding", [0] * len(row["stride"]))
    args = (row["stride"], row["padding"], row["dilation"], row["transposed"], op, row["groups"])
    for name, kern in (("port", False), ("kern", True)):
        g._KERNELS = kern

        def f() -> None:
            mx.eval([g._conv_core(xm, wm, bm, *args) for _ in range(reps)])

        out[name] = _median(f) / reps
    g._KERNELS = False
    g._WINO.clear()
    if mps:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        tdt = getattr(torch, dtype)
        xt = torch.from_numpy(x).to("mps", tdt)
        wt = torch.from_numpy(w).to("mps", tdt)
        bt = torch.from_numpy(bvec).to("mps", tdt) if bvec is not None else None
        nd = len(xs) - 2
        fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[nd]
        kw = {"stride": row["stride"], "padding": row["padding"], "dilation": row["dilation"], "groups": row["groups"]}
        if row["transposed"]:
            fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[nd]
            kw["output_padding"] = op

        def ft() -> None:
            with torch.inference_mode():
                for _ in range(reps):
                    fn(xt, wt, bt, **kw)
            torch.mps.synchronize()

        out["mps"] = _median(ft) / reps
    return out


def main() -> int:
    """Bench every row of the given shape files."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--no-mps", action="store_true")
    a = ap.parse_args()
    res = []
    for path in a.files:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        tot = {"port": 0.0, "kern": 0.0, "mps": 0.0}
        print(f"== {d['ref']} {d['input']}")
        for row in sorted(d["convs"], key=lambda r: -r["count"] * _flops(r)):
            t = bench(row, a.dtype, a.reps, mps=not a.no_mps)
            for k, v in t.items():
                tot[k] += v * row["count"]
            ratio = t["port"] / t["mps"] if "mps" in t else float("nan")
            print(
                f"{row['x']!s:24} w{row['w']!s:20} s{row['stride']} g{row['groups']} T{int(row['transposed'])} "
                f"x{row['count']:<3} port {t['port']:7.3f} kern {t['kern']:7.3f} mps {t.get('mps', 0):7.3f} "
                f"p/m {ratio:4.2f} GF/s {_flops(row) / t['port'] / 1e6:6.0f}"
            )
            res.append({"ref": d["ref"], **row, **t})
        print(f"   total ms: port {tot['port']:.1f} kern {tot['kern']:.1f} mps {tot['mps']:.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def _flops(r: dict) -> float:
    """Multiply-adds x 2 of the conv (from the input size; output ~ input / stride)."""
    import math  # noqa: PLC0415

    sp = math.prod(r["x"][2:])
    if r["transposed"]:
        sp *= math.prod(r["stride"])
        cout = r["w"][1] * r["groups"]
        cin = r["w"][0] // r["groups"]
    else:
        sp //= math.prod(r["stride"])
        cout, cin = r["w"][0], r["w"][1]
    return 2.0 * r["x"][0] * sp * cout * cin * math.prod(r["w"][2:])


if __name__ == "__main__":
    raise SystemExit(main())
