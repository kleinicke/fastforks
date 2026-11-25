"""Which share of a half-precision forward silently runs in float32: MLX replay (`FF_MLX_DTYPE_REPORT=1`) or MPS.

One quick forward per ref, each in its own process, at the speed survey's input scaled by `--scale` (synthetic bench
inputs, `--views` views for multi-view models). Prints `fastforks.shared.mlx_dtype_report`'s report per graph: the
float32 share of the matmul / conv / SDPA FLOPs, the modules that run them in float32 and the float32 origins.
`--mps`: the eager portable forward on MPS instead, under a dispatch mode that counts the same ops' FLOPs by input
dtype and prints the Python frames of the first float32 call sites (deliberate islands included).
`--float32-casts`: the other direction, the float32 casts (`.float()`, `.to(torch.float32)`, a cast to a weight's
dtype) that the float32 twin's export drops, by call site: a float32 stage of the PyTorch runner listed there runs in
the replay's dtype unless it uses `to_float32` / `float32_call` / `keep_float32`.

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


def _watch_float32_casts() -> dict[str, list]:
    """Count the float32 casts of float32 values during an MLX export, by fastforks call site.

    The export runs a float32 twin: such a cast records nothing, so what it starts runs in the replay's dtype. Most
    are harmless (a cast of the result, of an input, to a weight's dtype); a float32 stage the PyTorch runner keeps on
    purpose needs `to_float32` / `float32_call` / `keep_float32` (`fastforks._core.mlx_cast`).
    """
    import linecache  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from fastforks._core.tracing import exporting_for_mlx  # noqa: PLC0415

    sites: dict[str, list] = {}
    skip = ("mlx_cast.py", "mlx_graph.py", "mlx_wrap.py")

    def note(x: torch.Tensor) -> None:
        if not exporting_for_mlx() or x.dtype != torch.float32:
            return
        f = sys._getframe(2)  # noqa: SLF001
        while f is not None and ("/fastforks/" not in f.f_code.co_filename or f.f_code.co_filename.endswith(skip)):
            f = f.f_back
        if f is None:
            return
        text = linecache.getline(f.f_code.co_filename, f.f_lineno).strip()
        literal = any(s in text for s in ("float()", "float32", "FloatTensor"))
        key = f"{f.f_code.co_filename.split('/fastforks/', 1)[1]}:{f.f_lineno}"
        sites.setdefault(key, [0, literal, text[:110]])[0] += 1

    to, float_, type_ = torch.Tensor.to, torch.Tensor.float, torch.Tensor.type

    def patched_float(self, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        note(self)
        return float_(self, *a, **k)

    def patched_to(self, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        if torch.float32 in (*a, k.get("dtype")) and not any(isinstance(v, torch.Tensor) for v in a):
            note(self)
        return to(self, *a, **k)

    def patched_type(self, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        if (a[0] if a else k.get("dtype")) in (torch.float32, torch.FloatTensor, "torch.FloatTensor"):
            note(self)
        return type_(self, *a, **k)

    torch.Tensor.float, torch.Tensor.to, torch.Tensor.type = patched_float, patched_to, patched_type
    return sites


def worker(spec: dict[str, object]) -> None:
    import torch  # noqa: PLC0415

    from fastforks._core.bench import BenchSize, bench_inputs  # noqa: PLC0415

    task = __import__(f"fastforks.{spec['task']}", fromlist=["load"])
    mps = bool(spec.get("mps"))
    casts = _watch_float32_casts() if spec.get("casts") else None
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
    if casts is not None:
        print(f"float32 casts that vanish in the export: {len(casts)} call sites", flush=True)
        for site, (n, literal, text) in sorted(casts.items(), key=lambda kv: (not kv[1][1], kv[0])):
            print(f"  {'literal ' if literal else 'to dtype'} x{n:<3} {site}  | {text}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("refs", nargs="*")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--scale", type=float, default=0.5, help="of the survey input (default 0.5)")
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--mps", action="store_true", help="the eager MPS forward instead of the MLX replay")
    p.add_argument(
        "--float32-casts",
        action="store_true",
        help="also list the float32 casts the MLX export drops (a float32 stage that runs in the replay's dtype)",
    )
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
        spec["casts"] = a.float32_casts and not a.mps
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(spec)], env=env, check=False)
        if r.returncode:
            print(f"== {ref}: exit {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
