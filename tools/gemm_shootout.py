"""Fused ViT-block GEMMs: every `dpt_fused` tile configuration against cuBLAS / cuBLASLt, one GPU.

    uv run python tools/gemm_shootout.py [--m 20195 577] [--out rows.json]
    uvx modal run tools/modal_survey.py::shootout --gpu L4      # the same on a Modal GPU

Shapes: depth-pro's ViT-L blocks (35 patches + 1 image of 577 tokens, d 1024): qkv (bias), proj and fc2 (bias,
layer scale, residual), fc1 (bias, erf GELU). Variants, all fp16 in / out:

- `ff<cfg>`: `dpt_fused` with that tile configuration (0-4 fp16 accumulation, 5-6 fp32), epilogue fused;
  `ff_rule` is what `gemm_cfg` picks today
- `cublas_unfused`: the portable path, `F.linear` + separate GELU / `res + g * y` (fp32 accumulation)
- `cublas_only`: the bare GEMM `x @ w.T`, no epilogue (the ceiling)
- `lt_epi`: cuBLASLt epilogues through torch: `_addmm_activation(use_gelu)` (tanh GELU) for fc1, `addmm(bias)` for
  qkv, `addmm(res)` (beta 1, bias and scale folded into W / residual: not counted) for proj / fc2

Timing: CUDA events, rounds interleaved over the variants (the same clocks for all), median of the rounds.
Error: max |out - fp32 reference| (erf GELU), so a variant that is wrong or a different function shows up.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.kernels.dpt_fused import ACT_GELU, ACT_NONE, load

if TYPE_CHECKING:
    from collections.abc import Callable

D, HID = 1024, 4096
ROUNDS, ITERS, WARM = 7, 10, 3


def _shapes(m: int) -> list[dict]:
    return [
        {"name": "qkv", "m": m, "n": 3 * D, "k": D, "act": ACT_NONE, "res": False},
        {"name": "proj", "m": m, "n": D, "k": D, "act": ACT_NONE, "res": True},
        {"name": "fc1", "m": m, "n": HID, "k": D, "act": ACT_GELU, "res": False},
        {"name": "fc2", "m": m, "n": D, "k": HID, "act": ACT_NONE, "res": True},
    ]


def _variants(k, s: dict) -> dict[str, Callable[[], torch.Tensor]]:  # noqa: ANN001, C901 - DptFused; one closure per variant
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    m, n, kk = s["m"], s["n"], s["k"]
    x = torch.randn(m, kk, device=dev, generator=g).half()
    w = (torch.randn(n, kk, device=dev, generator=g) * kk**-0.5).half()
    b = (torch.randn(n, device=dev, generator=g) * 0.1).half()
    gs = (torch.rand(n, device=dev, generator=g) * 0.5 + 0.5) if s["res"] else None  # fp32 layer scale
    res = torch.randn(m, n, device=dev, generator=g).half() if s["res"] else None
    out = torch.empty(m, n, device=dev, dtype=torch.half)
    act = s["act"]

    y = x.float() @ w.float().T + b.float()
    if act == ACT_GELU:
        y = F.gelu(y)
    if s["res"]:
        y = res.float() + gs * y
    s["ref"] = y

    def ff(cfg: int | None) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            e = k._epi(m, n, out, b, gs, res, None, None, act)  # noqa: SLF001 - the library's own argument packing
            c = k.gemm_cfg(m, n, kk, act) if cfg is None else cfg
            st = torch.cuda.current_stream().cuda_stream
            k._check(k.lib.dpt_gemm(c, m, n, kk, x.data_ptr(), kk, w.data_ptr(), e, st), "gemm")  # noqa: SLF001
            return out

        return run

    vs: dict[str, Callable[[], torch.Tensor]] = {"ff_rule": ff(None)}
    vs |= {f"ff{c}": ff(c) for c in range(7)}
    gh = gs.half() if gs is not None else None

    def unfused() -> torch.Tensor:
        t = F.linear(x, w, b)
        if act == ACT_GELU:
            t = F.gelu(t)
        if s["res"]:
            t = res + gh * t
        return t

    wt = w.T
    lt_w = (w.float() * gs[:, None]).half().T if s["res"] else None  # layer scale folded into W

    def lt() -> torch.Tensor:
        if act == ACT_GELU:
            return torch._addmm_activation(b, x, wt, use_gelu=True)  # noqa: SLF001 - torch's cuBLASLt GELU_BIAS path
        if s["res"]:
            return torch.addmm(res, x, lt_w)  # bias folded into the residual: not counted
        return torch.addmm(b, x, wt)

    vs |= {"cublas_unfused": unfused, "cublas_only": lambda: x @ wt, "lt_epi": lt}
    return vs


def bench(ms: list[int]) -> dict:  # noqa: C901 - measure loop, flat on purpose
    k = load(torch.device("cuda"))
    p = torch.cuda.get_device_properties(0)
    rows = []
    for m in ms:
        for s in _shapes(m):
            vs = _variants(k, s)
            err = {}
            for name, f in vs.items():
                try:
                    err[name] = round((f().float() - s["ref"]).abs().max().item(), 4)
                except ValueError as exc:  # a configuration this shape cannot run
                    err[name] = str(exc)
            ok = {n: f for n, f in vs.items() if not isinstance(err[n], str)}
            times: dict[str, list[float]] = {n: [] for n in ok}
            for _ in range(ROUNDS):
                for name, f in ok.items():
                    for _ in range(WARM):
                        f()
                    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(ITERS):
                        f()
                    z.record()
                    torch.cuda.synchronize()
                    times[name].append(a.elapsed_time(z) / ITERS)
            flop = 2 * s["m"] * s["n"] * s["k"]
            row = {
                "shape": s["name"],
                "m": s["m"],
                "n": s["n"],
                "k": s["k"],
                "rule_cfg": k.gemm_cfg(s["m"], s["n"], s["k"], s["act"]),
            }
            for name in vs:
                if name in times:
                    t = statistics.median(times[name])
                    row[name] = {"ms": round(t, 4), "tflops": round(flop / t / 1e9, 1), "max_err": err[name]}
                else:
                    row[name] = {"error": err[name]}
            rows.append(row)
            print(json.dumps({r: row[r] for r in ("shape", "m")} | {n: row[n].get("tflops") for n in vs}), flush=True)
    return {"gpu": p.name, "sms": p.multi_processor_count, "torch": str(torch.__version__), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--m", type=int, nargs="+", default=[20195, 577], help="token counts (GEMM M)")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args()
    r = bench(a.m)
    if a.out:
        with open(a.out, "w") as f:  # noqa: PTH123
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
