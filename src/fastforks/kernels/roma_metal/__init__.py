"""Float32 RoMa v2 local correlation for PyTorch MPS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

load = metal_library("roma_metal", Path(__file__).with_name("correlation.metal"))


def correlate(lib: MetalLibrary, a: torch.Tensor, b: torch.Tensor, warp: torch.Tensor, radius: int) -> torch.Tensor:
    """BHWC features and normalised coordinates to a BCHW correlation volume in row-major offset order."""
    n, h, w, c = a.shape
    offsets = (2 * radius + 1) ** 2
    out = a.new_empty((n, h, w, offsets))
    lib.kernel("correlate", torch.float32)(
        a.contiguous(),
        b.contiguous(),
        warp.contiguous(),
        out,
        h,
        w,
        c,
        radius,
        threads=n * h * w * offsets * 32,
        group_size=256,
    )
    return out.permute(0, 3, 1, 2)
