"""Per-call sweep of the shared CUTLASS conv (`cnn_kernels.cutlass`): every tile config vs cuDNN, one forward.

Every distinct `cutlass(cfg, ...)` call of a kernels-mode forward (GRU zr / q convs, `conv_fused` convs) is timed
with tile configs 0-7, cuDNN channels-last `F.conv2d` (fp32 accumulate, conv only, no epilogue) and `cudnn_hacc`
(fp16 accumulate, conv only). Output: per call shape the median ms of each variant, the config the rule picked, and
the call count; plus forward totals (rule vs best config per shape).

    uv run python tools/conv_sweep.py --model crestereo-eth3d --out convs.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

import fastforks
from fastforks.kernels.cudnn_hacc import conv_half_acc
from fastforks.shared import cnn_kernels

if TYPE_CHECKING:
    from collections.abc import Callable

ROUNDS, ITERS, WARM = 5, 5, 2
CFGS = range(8)


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
    return {n: round(statistics.median(t), 4) for n, t in times.items()}


class Sweep:
    """Wraps `cutlass`; times each new call signature over the variants (the call itself then runs as usual)."""

    def __init__(self, orig: Callable[..., bool]) -> None:
        """Wrap `orig` (`cnn_kernels.cutlass`)."""
        self.orig, self.rows, self.order = orig, {}, []

    def __call__(self, cfg, x, w, hw, e, padding=None, stride=1) -> bool:  # noqa: ANN001, PLR0913, PLR0917 - wraps `cutlass`
        """Time the variants on a new signature, then run the call as given."""
        kk, c, r, s = w.shape
        pad = padding if padding is not None else (r // 2, s // 2)
        key = f"{hw[0]}x{hw[1]} {c}->{kk} {r}x{s} s{stride} mode{e.mode} act{e.act} rule{cfg}"
        if key in self.rows:
            self.rows[key]["calls"] += 1
            return self.orig(cfg, x, w, hw, e, padding=padding, stride=stride)
        torch.cuda.synchronize()
        try:
            x4 = x if x.dim() == 4 else x.view(1, *hw, c).permute(0, 3, 1, 2)  # noqa: PLR2004 - [1, C, H, W] view
        except RuntimeError:
            x4 = None  # a strided slice: no cuDNN reference

        def cut(k: int) -> Callable[[], bool]:
            return lambda: self.orig(k, x, w, hw, e, padding=padding, stride=stride)

        vs: dict[str, Callable[[], Any]] = {}
        ok: dict[int, bool] = {}
        for k in CFGS:
            ok[k] = bool(cut(k)())
            if ok[k]:
                vs[f"cfg{k}"] = cut(k)
        if x4 is not None:
            vs["cudnn_f32acc"] = lambda: F.conv2d(x4, w, None, stride, pad)
            try:
                conv_half_acc(x4, w, None, list(pad), [stride, stride])
                vs["cudnn_f16acc"] = lambda: conv_half_acc(x4, w, None, list(pad), [stride, stride])
            except RuntimeError:
                pass
        t = _time(vs)
        best = min((v, n) for n, v in t.items() if n.startswith("cfg"))
        self.rows[key] = {"calls": 1, "rule": cfg, "rule_ms": t.get(f"cfg{cfg}"), "best": best[1], "best_ms": best[0],
                          "pixels": hw[0] * hw[1] // (stride * stride), "c": c, "k": kk, "rs": f"{r}x{s}",
                          "unsupported": [k for k in CFGS if not ok[k]], "ms": t}  # fmt: skip
        self.order.append(key)
        return self.orig(cfg, x, w, hw, e, padding=padding, stride=stride)


def _summary(rows: list[dict]) -> dict[str, float]:
    tot = {"rule": 0.0, "best_cfg": 0.0, "cudnn_f32acc_conv_only": 0.0}
    per_cfg = {f"cfg{k}": 0.0 for k in CFGS}
    for r in rows:
        n = r["calls"]
        tot["rule"] += n * (r["rule_ms"] or 0)
        tot["best_cfg"] += n * r["best_ms"]
        tot["cudnn_f32acc_conv_only"] += n * r["ms"].get("cudnn_f32acc", r["rule_ms"] or 0)
        for k in per_cfg:  # one fixed config everywhere (rule config where unsupported)
            per_cfg[k] += n * r["ms"].get(k, r["rule_ms"] or 0)
    return {k: round(v, 2) for k, v in (tot | per_cfg).items()}


def run(model: str, size: tuple[int, int]) -> dict:
    orig = cnn_kernels.cutlass
    sw = Sweep(orig)
    m = fastforks.load(model, mode="kernels", device="cuda", dtype=torch.float16)
    imgs = [torch.rand(3, *size, device="cuda") for _ in range(2)]
    args = imgs if _wants_pair(m) else imgs[:1]
    m(*args)  # warm (kernel loads, cuDNN plans)
    torch.cuda.synchronize()
    mods = [mod for mod in list(sys.modules.values()) if getattr(mod, "cutlass", None) is orig]
    for mod in mods:
        mod.cutlass = sw
    try:
        m(*args)
    finally:
        for mod in mods:
            mod.cutlass = orig
    rows = [sw.rows[k] | {"key": k} for k in sw.order]
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "sms": p.multi_processor_count, "torch": str(torch.__version__), "model": model,
            "patched": [mod.__name__ for mod in mods], "summary": _summary(rows), "rows": rows}  # fmt: skip


def _wants_pair(m: Any) -> bool:  # noqa: ANN401 - any loaded model
    import inspect  # noqa: PLC0415

    try:
        return len(inspect.signature(m.__call__).parameters) >= 2  # noqa: PLR2004 - left, right
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", default="crestereo-eth3d")
    ap.add_argument("--size", type=int, nargs=2, default=[994, 1482], help="input height width")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args()
    random.seed(0)
    r = run(a.model, tuple(a.size))
    print(json.dumps(r["summary"], indent=1))
    nan = float("nan")
    for row in r["rows"]:
        f32, f16 = row["ms"].get("cudnn_f32acc", nan), row["ms"].get("cudnn_f16acc", nan)
        print(f"{row['key']:44s} x{row['calls']:<4d} best {row['best']} {row['best_ms']:.3f} rule {row['rule_ms']}"
              f" f32acc {f32:.3f} f16acc {f16:.3f}")  # fmt: skip
    if a.out:
        with open(a.out, "w") as f:  # noqa: PTH123
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
