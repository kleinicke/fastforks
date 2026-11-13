"""Shared parts of the Swin super-resolution networks (SwinIR, HAT).

`DropPath`, `to_2tuple`, `trunc_normal_` replace the timm / BasicSR helpers the vendored networks import (inference
only). `shift_mask` is SwinIR's `SwinTransformerBlock.calculate_mask` (the 0 / -100 shifted-window mask), made on the
device once per size. `relative_bias` gathers a block's relative position bias once, `window_attention` is the window
attention as one `scaled_dot_product_attention` call with that bias (+ the mask) as its additive mask.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["DropPath", "relative_bias", "shift_mask", "to_2tuple", "trunc_normal_", "window_attention"]

trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """`to_2tuple` of timm, for ints and pairs."""
    return x if isinstance(x, tuple) else (x, x)


def DropPath(drop_prob: float) -> nn.Module:  # noqa: N802 - the timm class name the vendored nets call
    """`DropPath` of timm at inference: the identity (stochastic depth only acts in training)."""
    del drop_prob
    return nn.Identity()


@functools.lru_cache(maxsize=16)
def shift_mask(h: int, w: int, window: int, shift: int, device: torch.device) -> torch.Tensor:
    """`[nW, window**2, window**2]` float32 0 / -100 mask of the shifted windows of an `h` x `w` map (SwinIR's)."""
    img = torch.zeros(1, h, w, 1, device=device)
    cnt = 0
    for hs in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for ws in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            img[:, hs, ws, :] = cnt
            cnt += 1
    m = img.view(1, h // window, window, w // window, window, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window)
    m = m.unsqueeze(1) - m.unsqueeze(2)
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)


def relative_bias(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """`[nH, N, N]` relative position bias of one block (`table[index]`, as the attention gathers it per call)."""
    n = index.shape[0]
    return table[index.reshape(-1)].view(n, n, -1).permute(2, 0, 1).contiguous()


def window_attention(  # noqa: PLR0913 - the block's projections, head count, scale, bias and mask
    x: torch.Tensor,
    qkv: nn.Module,
    proj: nn.Module,
    *,
    heads: int,
    scale: float,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Swin window attention of `[B * nW, N, C]` windows: `proj(softmax(q k^T * scale + bias (+ mask)) v)`.

    `bias` is `relative_bias(...)` `[nH, N, N]`, `mask` the `[nW, N, N]` shifted-window mask or None.
    """
    b_, n, c = x.shape
    q, k, v = qkv(x).reshape(b_, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4).unbind(0)
    add = bias.to(q.dtype)[None]
    if mask is not None:
        nw = mask.shape[0]
        add = add + mask.to(q.dtype)[:, None]
        if b_ != nw:
            add = add.repeat(b_ // nw, 1, 1, 1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=add.expand(b_, -1, -1, -1), scale=scale)
    return proj(out.transpose(1, 2).reshape(b_, n, c))
