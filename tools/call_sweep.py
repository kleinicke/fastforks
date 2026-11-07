"""Each `dpt_fused` GEMM / conv and flash-attention call of a forward: all tile configs vs cuBLAS / cuDNN / SDPA.

    uv run python tools/call_sweep.py [--model depth-pro] [--out calls.json]
    uvx modal run tools/modal_survey.py::sweep --gpu L4 --out calls.json     # the same on a Modal GPU

Runs one forward of the model in `kernels` mode with `DptFused.gemm`, `DptFused.conv` and the ViTs' `_flash` wrapped.
Each new call signature (shapes + epilogue) is timed before the real call runs, on scratch outputs (an in-place
residual stays intact), then the real call runs. Variants:

- `ff<cfg>`: `dpt_fused` with that tile configuration (GEMM 0-4 fp16 accumulation, 5-6 fp32; conv 0-5 fp16, 6-7
  fp32); `rule` is the configuration the library picks today
- `torch`: what the portable path does: `F.linear` / channels-last `F.conv2d` (fp32 accumulation) plus separate
  epilogue ops (and the pixel-shuffle copy); `torch_bench` the same with `cudnn.benchmark`; `deconv` (shuffle GEMMs)
  cuDNN `conv_transpose2d`
- attention: `flash64` (today) against `sdpa` (`F.scaled_dot_product_attention`)

Timing: CUDA events, rounds over the variants in random order, median. `err`: max |out - rule out|, so a variant that
computes something else shows up (fp16 against fp32 accumulation differ by ~1e-2).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

import fastforks
from fastforks.depth.models.depth_anything_v2 import fused as da2_fused
from fastforks.kernels.dpt_fused import _I, ACT_GELU, ACT_NONE, ACT_RELU, DptFused, _rows

if TYPE_CHECKING:
    from collections.abc import Callable

ROUNDS, ITERS, WARM = 5, 5, 2
GEMM_CFGS, CONV_CFGS = range(7), range(8)


def _time(vs: dict[str, Callable[[], Any]]) -> dict[str, float]:
    times: dict[str, list[float]] = {n: [] for n in vs}
    names = list(vs)
    for _ in range(ROUNDS):
        random.shuffle(names)
        for name in names:
            f = vs[name]
            for _ in range(WARM):
                f()
            a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(ITERS):
                f()
            z.record()
            torch.cuda.synchronize()
            times[name].append(a.elapsed_time(z) / ITERS)
    return {n: statistics.median(t) for n, t in times.items()}


def _act(t: torch.Tensor, act: int) -> torch.Tensor:
    return F.relu(t) if act == ACT_RELU else F.gelu(t) if act == ACT_GELU else t


def _epilogue(t: torch.Tensor, m: int, n: int, c: dict) -> torch.Tensor:
    """act((t + bias) * scale + res + res2) on the [m, n] rows (bias already added by the GEMM / conv), unfused."""
    t = t.reshape(m, n)
    if c["scale"] is not None:
        t = t * c["scale"].half()
    for r in (c["res"], c["res2"]):
        if r is not None:
            t = t + r.reshape(m, n)
    t = _act(t, c["act"])
    if c.get("y2") is not None:
        F.relu(t)
    return t


class Sweep:
    """Wrappers for the library calls; `rows` holds one timed entry per call signature (with its count)."""

    def __init__(self) -> None:
        """No calls seen yet."""
        self.rows: dict[str, dict] = {}
        self.order: list[str] = []

    def _record(self, key: str, info: dict, vs: dict[str, Callable[[], torch.Tensor]], rule: str) -> None:
        if key in self.rows:
            self.rows[key]["count"] += 1
            return
        out, err, ok = {}, {}, {}
        for name, f in vs.items():
            try:
                out[name] = f()
                ok[name] = f
            except (ValueError, RuntimeError) as exc:  # a configuration this shape cannot run
                err[name] = str(exc)[:80]
        torch.cuda.synchronize()
        ref = out[rule].float().reshape(-1)
        for name, y in out.items():
            if name != "deconv":  # NCHW / bias layout differs: timing only
                err[name] = round((y.float().reshape(-1) - ref).abs().max().item(), 4)
        ms = _time(ok)
        row = info | {"count": 1, "rule": rule, "ms": {n: round(t, 4) for n, t in ms.items()}, "err": err}
        ff = {n: t for n, t in ms.items() if n.startswith("ff")}
        row["best_ff"] = min(ff, key=ff.get) if ff else None
        row["best"] = min(ms, key=ms.get)
        self.rows[key] = row
        self.order.append(key)
        print(json.dumps({"key": key, "rule": rule, "best": row["best"], "ms": row["ms"]}), flush=True)

    def gemm(self, orig: Callable, k: DptFused, x: torch.Tensor, w: torch.Tensor, bias=None, **kw) -> torch.Tensor:  # noqa: ANN001, ANN003 - wraps the library's signature
        """Time `DptFused.gemm`'s call (every cfg, cuBLAS, cuDNN deconv for shuffles), then run it."""
        act, scale, res = kw.get("act", ACT_NONE), kw.get("scale"), kw.get("res")
        shuffle, acc32 = kw.get("shuffle"), kw.get("acc32", False)
        kk = x.shape[-1]
        m, n = x.numel() // kk, w.shape[0]
        lda = _rows(x, m, kk, "x")
        terms = f"b{int(bias is not None)} s{int(scale is not None)} r{int(res is not None)}"
        key = f"gemm m{m} n{n} k{kk} act{act} {terms}"
        key += f" sh{shuffle}" if shuffle else ""
        key += " acc32" if acc32 else ""
        if key in self.rows:
            self.rows[key]["count"] += 1
            return orig(k, x, w, bias, **kw)
        if shuffle is None:
            y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.half)
        else:
            gh, gw, sh, sw = shuffle
            cout, b = n // (sh * sw), m // (gh * gw)
            y = torch.empty((b, gh * sh, gw * sw, cout), device=x.device, dtype=torch.half)
        rule_cfg = (5 if -(-m // 128) * -(-n // 64) >= k.sms else 6) if acc32 else k.gemm_cfg(m, n, act)

        def ff(cfg: int) -> Callable[[], torch.Tensor]:
            def run() -> torch.Tensor:
                e = k._epi(m, n, y, bias, scale, res, None, None, act)  # noqa: SLF001 - the library's packing
                if shuffle is not None:
                    e.ldy, e.sh, e.sw, e.gh, e.gw, e.cout = cout, sh, sw, gh, gw, cout
                st = torch.cuda.current_stream().cuda_stream
                k._check(k.lib.dpt_gemm(cfg, m, n, kk, x.data_ptr(), lda, w.data_ptr(), e, st), "gemm")  # noqa: SLF001
                return y

            return run

        c = {"act": act, "scale": scale, "res": res, "res2": None}
        x2 = x.reshape(m, kk)

        def tch() -> torch.Tensor:
            t = _epilogue(F.linear(x2, w, bias), m, n, c)
            if shuffle is not None:
                t = t.view(b, gh, gw, sh, sw, cout).permute(0, 1, 3, 2, 4, 5).contiguous()
            return t

        vs = {f"ff{i}": ff(i) for i in GEMM_CFGS} | {"torch": tch}
        if shuffle is not None:
            wd = w.view(sh, sw, cout, kk).permute(3, 2, 0, 1).contiguous(memory_format=torch.channels_last)
            bd = bias.view(sh * sw, cout)[0].contiguous() if bias is not None else None
            xd = x2.view(b, gh, gw, kk).permute(0, 3, 1, 2)
            vs["deconv"] = lambda: _act(F.conv_transpose2d(xd, wd, bd, stride=(sh, sw)), act)
        info = {"kind": "shuffle" if shuffle else "gemm", "m": m, "n": n, "k": kk, "act": act, "flop": 2 * m * n * kk}
        self._record(key, info, vs, f"ff{rule_cfg}")
        return orig(k, x, w, bias, **kw)

    def conv(self, orig: Callable, k: DptFused, x: torch.Tensor, w: torch.Tensor, bias=None, **kw) -> torch.Tensor:  # noqa: ANN001, ANN003 - wraps the library's signature
        """Time `DptFused.conv`'s call (every cfg, cuDNN), then run it."""
        pad, stride, act = kw.get("padding", 0), kw.get("stride", 1), kw.get("act", ACT_NONE)
        res, res2, y2 = kw.get("res"), kw.get("res2"), kw.get("y2")
        b, h, w_, cc = x.shape
        kk, r, s, _ = w.shape
        p, q = (h + 2 * pad - r) // stride + 1, (w_ + 2 * pad - s) // stride + 1
        m = b * p * q
        key = f"conv {b}x{h}x{w_}x{cc} -> {kk} {r}x{s} p{pad} s{stride} act{act} b{int(bias is not None)}"
        key += f" r{int(res is not None)} r2{int(res2 is not None)} y2{int(y2 is not None)}"
        if key in self.rows:
            self.rows[key]["count"] += 1
            return orig(k, x, w, bias, **kw)
        y = torch.empty((b, p, q, kk), device=x.device, dtype=torch.half)
        ys = torch.empty_like(y) if y2 is not None else None

        def ff(cfg: int) -> Callable[[], torch.Tensor]:
            def run() -> torch.Tensor:
                e = k._epi(m, kk, y, bias, None, res, res2, ys, act)  # noqa: SLF001 - the library's packing
                g = (_I * 11)(b, h, w_, cc, kk, r, s, pad, pad, stride, stride)
                st = torch.cuda.current_stream().cuda_stream
                k._check(k.lib.dpt_conv(cfg, x.data_ptr(), w.data_ptr(), g, e, st), "conv")  # noqa: SLF001
                return y

            return run

        xc, wc = x.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2)
        c = {"act": act, "scale": None, "res": res, "res2": res2, "y2": y2}

        def tch() -> torch.Tensor:
            return _epilogue(F.conv2d(xc, wc, bias, stride, pad).permute(0, 2, 3, 1), m, kk, c)

        def tch_bench() -> torch.Tensor:
            torch.backends.cudnn.benchmark = True
            try:
                return tch()
            finally:
                torch.backends.cudnn.benchmark = False

        vs = {f"ff{i}": ff(i) for i in CONV_CFGS} | {"torch": tch, "torch_bench": tch_bench}
        info = {"kind": "conv", "m": m, "n": kk, "k": cc * r * s, "act": act, "flop": 2 * m * kk * cc * r * s}
        self._record(key, info, vs, f"ff{k.conv_cfg(m, kk)}")
        return orig(k, x, w, bias, **kw)

    def flash(self, orig: Callable, qkv: torch.Tensor, scale: float, attn: Any) -> torch.Tensor:  # noqa: ANN401 - Flash64
        """Time `_flash` (flash64 vs SDPA), then run it."""
        b, n, _, h, d = qkv.shape
        key = f"attn b{b} n{n} h{h} d{d}"
        if key not in self.rows:
            q, kk, v = (t.transpose(1, 2) for t in qkv.unbind(2))

            def sdpa() -> torch.Tensor:
                o = F.scaled_dot_product_attention(q, kk, v, scale=scale)
                return o.transpose(1, 2).reshape(b, n, h * d)

            info = {"kind": "attn", "m": b * n, "n": n, "k": d, "act": 0, "flop": 4 * b * h * n * n * d}
            self._record(key, info, {"flash64": lambda: orig(qkv, scale, attn), "sdpa": sdpa}, "flash64")
        else:
            self.rows[key]["count"] += 1
        return orig(qkv, scale, attn)


def _summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    """GPU ms per kind and in all: today's rule, the best `dpt_fused` cfg, torch, and the best of everything."""
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        ms, cnt = r["ms"], r["count"]
        torch_ms = min((t for n, t in ms.items() if not n.startswith("ff") and n != "flash64"), default=float("nan"))
        best_ff = ms[r["best_ff"]] if r["best_ff"] else ms[r["rule"]]
        vals = {"rule": ms[r["rule"]], "best_ff": best_ff, "torch": torch_ms, "best": ms[r["best"]]}
        for kind in (r["kind"], "all"):
            s = out.setdefault(kind, {"calls": 0} | dict.fromkeys(vals, 0.0))
            s["calls"] += cnt
            for name, v in vals.items():
                s[name] = round(s[name] + cnt * v, 2)
    return out


def run(model: str, size: tuple[int, int]) -> dict:
    sw = Sweep()
    g0, c0, f0 = DptFused.gemm, DptFused.conv, da2_fused._flash  # noqa: SLF001
    m = fastforks.load(model, mode="kernels", device="cuda", dtype=torch.float16)
    img = torch.rand(3, *size, device="cuda")
    m(img)  # warm (kernel loads, cuDNN plans)
    torch.cuda.synchronize()
    DptFused.gemm = lambda k, *a, **kw: sw.gemm(g0, k, *a, **kw)
    DptFused.conv = lambda k, *a, **kw: sw.conv(c0, k, *a, **kw)
    da2_fused._flash = lambda *a: sw.flash(f0, *a)  # noqa: SLF001
    try:
        m(img)
    finally:
        DptFused.gemm, DptFused.conv, da2_fused._flash = g0, c0, f0  # noqa: SLF001
    rows = [sw.rows[k] | {"key": k} for k in sw.order]
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "sms": p.multi_processor_count, "torch": str(torch.__version__), "model": model,
            "summary": _summary(rows), "rows": rows}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="depth-pro")
    ap.add_argument("--size", type=int, nargs=2, default=[994, 1482], help="input height width")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args()
    random.seed(0)
    r = run(a.model, tuple(a.size))
    print(json.dumps(r["summary"], indent=1))
    if a.out:
        with open(a.out, "w") as f:  # noqa: PTH123
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
