"""Which share of a half-precision forward silently runs in float32: MLX replay (`FF_MLX_DTYPE_REPORT=1`) or MPS.

One quick forward per ref, each in its own process, at the speed survey's input scaled by `--scale` (synthetic bench
inputs, `--views` views for multi-view models). Prints `fastforks.shared.mlx_dtype_report`'s report per graph: the
float32 share of the matmul / conv / SDPA FLOPs, the modules that run them in float32 and the float32 origins.
`--mps`: the eager portable forward on MPS instead, under a dispatch mode that counts the same ops' FLOPs by input
dtype and prints the Python frames of the first float32 call sites (deliberate islands included).

    PYTHONPATH=src uv run python tools/mlx_dtype_report.py pi3 vggt-1b --scale 0.5
    PYTHONPATH=src uv run python tools/mlx_dtype_report.py gmflow-scale1-things --mps
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent


def _job(ref: str) -> dict[str, str]:
    sys.path.insert(0, str(TOOLS))
    import speed_survey  # noqa: PLC0415

    jobs = speed_survey.plan(speed_survey._device("apple-m1-pro-mlx-16gb"), only=[ref], modes=["portable"])  # noqa: SLF001
    if not jobs:
        msg = f"{ref}: no MLX survey job"
        raise SystemExit(msg)
    return jobs[0]


_HEAVY = ("mm", "bmm", "addmm", "baddbmm", "convolution", "_scaled_dot_product", "scaled_dot_product", "linear")


def _site() -> str:
    """The innermost fastforks frames of the current call (not this tool's, not torch's)."""
    import traceback  # noqa: PLC0415

    frames = [
        f for f in traceback.extract_stack() if "/fastforks/" in f.filename and "mlx_dtype_report" not in f.filename
    ]
    return " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames[-3:]))


def _mps_report(model: object, args: tuple[object, ...], kw: dict[str, object]) -> None:
    import torch  # noqa: PLC0415
    from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415

    flops = {"f32": 0.0, "half": 0.0}
    sites: dict[str, list[float]] = {}

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, a=(), k=None):  # noqa: ANN001, ANN204
            out = func(*a, **(k or {}))
            name = func.__name__.split(".")[0]
            if name.lstrip("_") in {h.lstrip("_") for h in _HEAVY} or "scaled_dot_product" in name:
                xs = [x for x in (*a, *(k or {}).values()) if isinstance(x, torch.Tensor) and x.is_floating_point()]
                o = out[0] if isinstance(out, (tuple, list)) else out
                size = o.numel() if isinstance(o, torch.Tensor) else 0
                inner = xs[1].shape[-1] if "conv" not in name and len(xs) > 1 else 1
                if "conv" in name and len(xs) > 1:
                    inner = xs[1].numel() // max(xs[1].shape[0], 1)
                fl = 2.0 * size * inner
                if any(x.dtype == torch.float32 for x in xs):
                    flops["f32"] += fl
                    sites.setdefault(f"{name} {_site()}", [0.0, 0])[0] += fl
                    sites[f"{name} {_site()}"][1] += 1
                else:
                    flops["half"] += fl
            return out

    with torch.inference_mode(), Mode():
        model(*args, **kw)
    tot = sum(flops.values()) or 1.0
    print(
        f"mps dtype report: heavy FLOPs float32 {100 * flops['f32'] / tot:.1f} % of {tot / 1e9:.1f} GFLOP", flush=True
    )
    for site, (fl, n) in sorted(sites.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  float32 {100 * fl / tot:5.1f} %  x{n}  {site}", flush=True)


def worker(spec: dict[str, object]) -> None:
    import torch  # noqa: PLC0415

    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    task = __import__(f"fastforks.{spec['task']}", fromlist=["load"])
    mps = bool(spec.get("mps"))
    model = task.load(spec["ref"], mode="portable", device="mps" if mps else "mlx", dtype=spec["dtype"])
    h, w = (int(v) for v in str(spec["input"]).split("x"))
    s = float(spec["scale"])
    h, w = (max(64, round(v * s / 32) * 32) for v in (h, w))
    args, kw = bench_inputs(model, BenchSize(h, w, int(spec["views"])))
    print(f"== {spec['ref']} {spec['dtype']} {h}x{w}{' mps' if mps else ''}", flush=True)
    if mps:
        _mps_report(model, args, kw)
        return
    with torch.inference_mode():
        model(*args, **kw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("refs", nargs="*")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--scale", type=float, default=0.5, help="of the survey input (default 0.5)")
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--mps", action="store_true", help="the eager MPS forward instead of the MLX replay")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        worker(json.loads(a.worker))
        return
    env = {**os.environ, "FF_MLX_DTYPE_REPORT": "1", "FASTFORKS_OFFLINE": "1", "FASTFORKS_NONCOMMERCIAL": "1"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    for ref in a.refs:
        job = _job(ref)
        spec = {**job, "dtype": a.dtype, "scale": a.scale, "views": a.views, "mps": a.mps}
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(spec)], env=env, check=False)
        if r.returncode:
            print(f"== {ref}: exit {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
