"""Where a depth-pro `kernels` forward spends its wall time: sections, side streams, clocks / power, non-kernel ops.

    uv run python tools/section_time.py [--out sections.json]
    uvx modal run tools/modal_survey.py::sections --gpu L4 --out sections.json     # the same on a Modal GPU

Measures, all at 1536x1536 (the network input; the survey's 994x1482 image is resized to it):

- `e2e`: the whole model call, portable and kernels (streams on / off), back to back (sustained) and with 2 s idle
  gaps before each call (cold clocks), with the SM clock, power and temperature logged by `nvidia-smi` meanwhile;
- `sections`: encode / decode / head / FOV tail of the kernels net with a sync between them (streams on / off);
- `profile`: one kernels forward with streams off under the torch profiler: GPU time per kernel family, so the
  time outside the `dpt_fused` GEMM / conv and flash kernels shows up.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import time
from typing import TYPE_CHECKING, Any

import torch
from torch.profiler import ProfilerActivity, profile

import fastforks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SIZE = 1536
SUSTAINED, GAPPED, GAP_S, WARM = 20, 5, 2.0, 3


@contextlib.contextmanager
def _smi(log: dict) -> Iterator[None]:
    """Poll SM clock (MHz), power (W) and temperature (C) every 100 ms; their means go into `log`."""
    q = "--query-gpu=clocks.sm,power.draw,temperature.gpu"
    cmd = ["nvidia-smi", q, "--format=csv,noheader,nounits", "-lms", "100"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    try:
        yield
    finally:
        p.terminate()
        out = p.communicate()[0]
        vals = [[float(v) for v in ln.split(",")] for ln in out.splitlines() if ln.count(",") == 2]  # noqa: PLR2004
        vals = vals[2:] or vals  # the first samples predate the work
        if vals:
            for i, name in enumerate(("sm_mhz", "power_w", "temp_c")):
                col = [v[i] for v in vals]
                log[name] = round(statistics.mean(col), 1)
                log[name + "_min"] = round(min(col), 1)


def _ms(f: Callable[[], Any]) -> float:
    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    f()
    z.record()
    torch.cuda.synchronize()
    return a.elapsed_time(z)


def _e2e(f: Callable[[], Any]) -> dict:
    for _ in range(WARM):
        f()
    torch.cuda.synchronize()
    r: dict[str, Any] = {}
    with _smi(r):
        ts = [_ms(f) for _ in range(SUSTAINED)]
    r |= {"ms": round(statistics.median(ts), 1), "first": round(ts[0], 1), "last": round(ts[-1], 1)}
    gts = []
    for _ in range(GAPPED):
        time.sleep(GAP_S)
        gts.append(_ms(f))
    r["gapped_ms"] = round(statistics.median(gts), 1)
    return r


def _one_pass(net: Any, x: torch.Tensor) -> dict[str, float]:  # noqa: ANN401 - depth-pro's _KernelsNet
    """One forward, section by section (synced in between): ms of each."""
    st: dict[str, Any] = {}

    def fov() -> None:
        lr = net.fov_down(st["d"][1].permute(0, 3, 1, 2).float())
        net.fov_head(st["e"][1].t().reshape_as(lr) + lr)

    return {
        "encode": _ms(lambda: st.update(e=net._encode(x, with_fov=True))),  # noqa: SLF001
        "decode": _ms(lambda: st.update(d=net._decode(st["e"][0]))),  # noqa: SLF001
        "head": _ms(lambda: st.update(c=net._head(st["d"][0]))),  # noqa: SLF001
        "fov": _ms(fov),
    }


def _sections(net: Any, x: torch.Tensor) -> dict:  # noqa: ANN401 - depth-pro's _KernelsNet
    """Median ms of each section over 5 forwards."""
    runs = [_one_pass(net, x) for _ in range(WARM + 5)][WARM:]
    return {k: round(statistics.median(r[k] for r in runs), 2) for k in runs[0]}


def _family(name: str) -> str:
    n = name.lower()
    for fam, keys in (
        ("dpt_gemm", ("fused_gemm",)),
        ("dpt_conv", ("fused_conv",)),
        ("flash64", ("flash64", "flash_fwd", "fastbb")),
        ("layer_norm", ("layer_norm", "layernorm")),
        ("dpt_other", ("dpt_", "patches", "bilinear", "head_dot")),
        ("gemm_other", ("gemm", "sgemm", "cutlass", "ampere", "sm80", "sm89", "cublas")),
        ("copy_index", ("copy", "index", "gather", "cat", "fill")),
        ("elementwise", ("elementwise", "vectorized", "reduce")),
    ):
        if any(k in n for k in keys):
            return fam
    return "other"


def _profile(f: Callable[[], Any]) -> dict:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        f()
        torch.cuda.synchronize()
    fams: dict[str, float] = {}
    top = []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
        if us <= 0:
            continue
        fam = _family(e.key)
        fams[fam] = fams.get(fam, 0.0) + us / 1e3
        top.append((us / 1e3, e.count, fam, e.key[:90]))
    top.sort(reverse=True)
    return {
        "families_ms": {k: round(v, 2) for k, v in sorted(fams.items(), key=lambda kv: -kv[1])},
        "total_ms": round(sum(fams.values()), 1),
        "top": [{"ms": round(t, 2), "count": c, "family": fam, "kernel": k} for t, c, fam, k in top[:25]],
    }


@torch.inference_mode()
def run(model: str) -> dict:
    img = torch.rand(3, SIZE, SIZE, device="cuda")
    kern = fastforks.load(model, mode="kernels", device="cuda", dtype=torch.float16)
    port = fastforks.load(model, mode="portable", device="cuda", dtype=torch.float16)
    net = kern._runner.net  # noqa: SLF001 - depth-pro's _KernelsNet
    side = net.streams
    x = img[None].float().contiguous()
    r: dict[str, Any] = {"e2e": {}, "sections": {}}
    r["e2e"]["portable"] = _e2e(lambda: port(img))
    for name, st in (("kernels", side), ("kernels_no_streams", None)):
        net.streams = st
        r["e2e"][name] = _e2e(lambda: kern(img))
        r["sections"][name] = _sections(net, x)
        print(json.dumps({name: r["e2e"][name], "sections": r["sections"][name]}), flush=True)
    print(json.dumps({"portable": r["e2e"]["portable"]}), flush=True)
    net.streams = None
    r["profile_kernels_no_streams"] = _profile(lambda: net(x, with_fov=True))
    r["profile_portable"] = _profile(lambda: port(img))
    net.streams = side
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "sms": p.multi_processor_count, "torch": str(torch.__version__), "model": model} | r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="depth-pro")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args()
    r = run(a.model)
    for k in ("profile_kernels_no_streams", "profile_portable"):
        print(k, json.dumps(r[k]["families_ms"]), r[k]["total_ms"])
    if a.out:
        with open(a.out, "w") as f:  # noqa: PTH123
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
