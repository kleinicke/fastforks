"""Pillow-exact bilinear and bicubic resizes (8-bit images), bit-exact on any device."""

from __future__ import annotations

import functools
import math

import torch

PIL_BITS = 22  # fixed: Pillow `PRECISION_BITS` of the 8-bit resampling (32 - 8 - 2)


PIL_CUBIC = -0.5  # fixed: Pillow's bicubic filter parameter


def _filter(x: torch.Tensor, *, cubic: bool) -> torch.Tensor:
    """Pillow `bilinear_filter` / `bicubic_filter` (a = -0.5) of float64 offsets."""
    x = x.abs()
    if not cubic:
        return (1 - x).clamp(min=0)
    a = PIL_CUBIC
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = (((x - 5) * x + 8) * x - 4) * a
    return torch.where(x < 1, near, torch.where(x < 2, far, 0.0))  # noqa: PLR2004 - the cubic's support


@functools.lru_cache(maxsize=64)
def _pil_taps(n_in: int, n_out: int, device: torch.device, *, cubic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Pillow `precompute_coeffs` + `normalize_coeffs_8bpc`: source indices, int weights `[n_out, K]`."""
    scale = n_in / n_out
    fs = max(scale, 1.0)  # the filter widens when downscaling (antialiasing)
    support = (2.0 if cubic else 1.0) * fs
    t = torch.arange(math.ceil(support) * 2 + 1, dtype=torch.float64)
    center = (torch.arange(n_out, dtype=torch.float64) + 0.5) * scale
    lo = (center - support + 0.5).trunc().clamp(min=0)
    n = (center + support + 0.5).trunc().clamp(max=n_in) - lo
    t = t[: int(n.max())]  # Pillow's trailing taps are zero for every output: skip them
    w = _filter((t + lo[:, None] - center[:, None] + 0.5) * (1 / fs), cubic=cubic)
    w = torch.where(t < n[:, None], w, 0.0)
    w = w / w.sum(1, keepdim=True) * (1 << PIL_BITS)
    k = (w + torch.where(w < 0, -0.5, 0.5)).trunc().to(torch.int32)  # (int)(k * 2^22 +- 0.5)
    return (lo[:, None] + t).clamp(max=n_in - 1).to(device, torch.int64), k.to(device)


# data-dependent tap tables (lru_cache, int(max)): one clean graph break instead of dynamo tracing the loop
@torch.compiler.disable
def pil_bilinear(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pillow `Image.resize(BILINEAR)` of a uint8-valued `[B, C, H, W]` tensor, bit-exact on any device.

    Two passes in Pillow's fixed-point arithmetic, each rounded to uint8; Pillow's pass order (vertical first only
    when the height shrinks by more than twice the width's reduction).
    """
    return _pil_resize(x, size, cubic=False)


@torch.compiler.disable
def pil_bicubic(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Pillow `Image.resize(BICUBIC)` (torchvision's PIL `Resize` default for bicubic), as `pil_bilinear`."""
    return _pil_resize(x, size, cubic=True)


def _pil_resize(x: torch.Tensor, size: tuple[int, int], *, cubic: bool) -> torch.Tensor:
    (h, w), (nh, nw) = x.shape[-2:], size
    passes = [(-1, w, nw), (-2, h, nh)]
    if h - nh > max(0, 2 * (w - nw)):
        passes.reverse()
    x = x.to(torch.int32)
    for dim, n_in, n_out in passes:
        if n_in == n_out:
            continue
        idx, k = _pil_taps(n_in, n_out, x.device, cubic=cubic)
        k = k.view(*k.shape, *[1] * (-1 - dim))  # weights broadcast along the rows pass's width
        acc = x.index_select(dim, idx[:, 0]).mul_(k[:, 0]).add_(1 << (PIL_BITS - 1))
        for j in range(1, idx.shape[1]):
            acc.addcmul_(x.index_select(dim, idx[:, j]), k[:, j])
        x = acc.div_(1 << PIL_BITS, rounding_mode="floor").clamp_(0, 255)
    return x.float()
