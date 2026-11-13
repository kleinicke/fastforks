"""The long-side resize of upstream LightGlue's `ImagePreprocessor` (kornia `resize`, Apache-2.0, Kornia Team)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_long(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """The kornia `resize(image, long_side, side="long", antialias=True)` of `[B, C, H, W]`."""
    h, w = image.shape[-2:]
    ratio = w / h
    size = (long_side, int(long_side * ratio)) if ratio < 1.0 else (int(long_side / ratio), long_side)
    factors = (h / size[0], w / size[1])
    if max(factors) > 1:  # kornia's antialiasing: a Gaussian blur (reflect borders) before downscaling
        x = image
        for dim, f in ((-1, factors[1]), (-2, factors[0])):
            sigma = max((f - 1.0) / 2.0, 0.001)
            k = int(max(4.0 * sigma, 3))
            k += 1 - k % 2
            t = torch.arange(k, device=image.device, dtype=image.dtype) - k // 2
            g = torch.exp(-t.pow(2.0) / (2 * sigma**2))
            g = (g / g.sum()).expand(x.shape[1], 1, k).contiguous()
            if dim == -1:
                x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g[:, :, None], groups=x.shape[1])
            else:
                x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g[:, :, :, None], groups=x.shape[1])
        image = x
    return F.interpolate(image, size=size, mode="bilinear", align_corners=None)
