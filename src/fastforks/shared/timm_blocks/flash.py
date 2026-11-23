"""timm `Attention` through the `fastbb` flash kernel (the kernels paths of WAFT and WAFT-Stereo)."""

from __future__ import annotations

import torch
from torch import nn

from fastforks._core.patch import override
from fastforks.kernels.fastbb import Flash64

from .net.vit import Attention


def _attention_forward(self: Attention, x: torch.Tensor) -> torch.Tensor:
    """A timm `Attention` at eval (no q / k norm) with the `fastbb` flash kernel; the same math as SDPA."""
    b, n, c = x.shape
    flash: Flash64 = self.flash  # ty: ignore[invalid-assignment]  # set by `use_flash_attention`
    d = c // self.num_heads
    q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, d).unbind(2)
    return self.proj(flash(q, k, v, d**-0.5).view(b, n, c))  # [B, N, H, 64] views, head dim checked at setup


def use_flash_attention(model: nn.Module, variant: str = "hh") -> int:
    """Route every timm `Attention` in `model` (fp16, eval, head dim 64) through one `Flash64(variant)`.

    q, k and v are read as views of the packed qkv (SDPA's input needs a permute and its output a transpose copy).
    The `Flash64` probes its first launches and widens its accumulation if one overflows (see `fastbb.Flash64`).
    Returns the number of attention modules changed.

    Raises:
        TypeError: An attention has q / k norms.
        ValueError: An attention's head dim is not 64.
    """
    flash, n = Flash64(variant), 0
    for m in model.modules():
        if isinstance(m, Attention):
            if not (isinstance(m.q_norm, nn.Identity) and isinstance(m.k_norm, nn.Identity)):
                msg = "use_flash_attention: q / k norms are not implemented"
                raise TypeError(msg)
            if m.qkv.in_features != 64 * m.num_heads:
                msg = f"use_flash_attention: head dim {m.qkv.in_features // m.num_heads}, expected 64"
                raise ValueError(msg)
            m.flash = flash  # ty: ignore[invalid-assignment]  # a plain attribute, not a submodule
            override(m, _attention_forward)
            n += 1
    return n
