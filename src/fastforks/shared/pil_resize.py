"""Pillow-exact bilinear resize (8-bit images), bit-exact on any device."""

from __future__ import annotations

import functools
import math

import torch

PIL_BITS = 22  # fixed: Pillow `PRECISION_BITS` of the 8-bit resampling (32 - 8 - 2)


@functools.lru_cache(maxsize=64)
def _pil_taps(n_in: int, n_out: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` (bilinear): source indices, int weights `[n_out, K]`."""
    scale = n_in / n_out
    fs = max(scale, 1.0)  # the filter widens when downscaling (antialiasing); bilinear support 1
    t = torch.arange(math.ceil(fs) * 2 + 1, dtype=torch.float64)
    center = (torch.arange(n_out, dtype=torch.float64) + 0.5) * scale
    lo = (center - fs + 0.5).trunc().clamp(min=0)
    n = (center + fs + 0.5).trunc().clamp(max=n_in) - lo
    t = t[: int(n.max())]  # Pillow's trailing taps are zero for every output: skip them
    w = (1 - ((t + lo[:, None] - center[:, None] + 0.5) * (1 / fs)).abs()).clamp(min=0)
    w = torch.where(t < n[:, None], w, 0.0)
    k = (w / w.sum(1, keepdim=True) * (1 << PIL_BITS) + 0.5).trunc().to(torch.int32)
    return (lo[:, None] + t).clamp(max=n_in - 1).to(device, torch.int64), k.to(device)


# data-dependent tap tables (lru_cache, int(max)): one clean graph break instead of dynamo tracing the loop
@torch.compiler.disable
def pil_bilinear(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pillow `Image.resize(BILINEAR)` of a uint8-valued `[B, C, H, W]` tensor, bit-exact on any device.

    Two passes in Pillow's fixed-point arithmetic, each rounded to uint8; Pillow's pass order (vertical first only
    when the height shrinks by more than twice the width's reduction).
    """
    (h, w), (nh, nw) = x.shape[-2:], size
    passes = [(-1, w, nw), (-2, h, nh)]
    if h - nh > max(0, 2 * (w - nw)):
        passes.reverse()
    x = x.to(torch.int32)
    for dim, n_in, n_out in passes:
        if n_in == n_out:
            continue
        idx, k = _pil_taps(n_in, n_out, x.device)
        k = k.view(*k.shape, *[1] * (-1 - dim))  # weights broadcast along the rows pass's width
        acc = x.index_select(dim, idx[:, 0]).mul_(k[:, 0]).add_(1 << (PIL_BITS - 1))
        for j in range(1, idx.shape[1]):
            acc.addcmul_(x.index_select(dim, idx[:, j]), k[:, j])
        x = acc.div_(1 << PIL_BITS, rounding_mode="floor").clamp_(0, 255)
    return x.float()
