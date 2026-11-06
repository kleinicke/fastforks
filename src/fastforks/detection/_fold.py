"""BatchNorm folding shared by the detection families."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F

if TYPE_CHECKING:
    import torch


def fold_bn_into(sd: dict[str, torch.Tensor], bn: str, conv: str, eps: float) -> dict[str, torch.Tensor]:
    """`{conv}.weight` and `{conv}.bias` with BatchNorm `bn` folded in (only the bias if the weight does not fit)."""
    g, b, m, v = (sd[f"{bn}.{n}"].double() for n in ("weight", "bias", "running_mean", "running_var"))
    s = g * (v + eps).rsqrt()
    w = sd.get(f"{conv}.weight")
    if w is None or w.shape[:1] != s.shape:  # the strict load reports it
        return {f"{conv}.bias": (b - m * s).to(sd[f"{bn}.weight"].dtype if w is None else w.dtype)}
    return {f"{conv}.weight": (w.double() * s.view(-1, 1, 1, 1)).to(w.dtype), f"{conv}.bias": (b - m * s).to(w.dtype)}


def merge_repvgg(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold every RepVGG block (`...bottlenecks.N.conv1` 3x3 + `conv2` 1x1, BatchNorm already folded) into one `conv`.

    The 3x3 and 1x1 branches add up to one 3x3 convolution. Tensors that do not fit are left for the strict load.
    """
    out = dict(sd)
    for k in [k for k in out if k.endswith(".conv1.conv.weight") and ".bottlenecks." in k]:
        p = k.removesuffix("conv1.conv.weight")
        if f"{p}conv2.conv.weight" not in out or out[k].shape[:2] != out[f"{p}conv2.conv.weight"].shape[:2]:
            continue  # the strict load reports it
        out[f"{p}conv.weight"] = out.pop(k) + F.pad(out.pop(f"{p}conv2.conv.weight"), [1, 1, 1, 1])
        out[f"{p}conv.bias"] = out.pop(f"{p}conv1.conv.bias") + out.pop(f"{p}conv2.conv.bias")
    return out
