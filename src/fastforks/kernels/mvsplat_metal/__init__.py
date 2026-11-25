"""MVSplat plane-sweep correlation for PyTorch MPS: bilinear warp and channel dot product in one Metal kernel."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

load = metal_library("mvsplat_metal", Path(__file__).with_name("plane_sweep.metal"))


def accumulate(
    lib: MetalLibrary, features: torch.Tensor, grid: torch.Tensor, volume: torch.Tensor, offset: int
) -> None:
    """`mvsplat_corr.accumulate` on MPS: view `n` against view `(n + offset) % V`; offset 1 writes, later ones add.

    `features` `[V, C, h, w]` channels-last (fp16 or fp32, `C % 4 == 0`), `grid` `[V, D, h*w, 2]` float32
    (`grid_sample` coordinates, align_corners, zeros padding), `volume` `[V, D, h, w]` float32 (unscaled dot products).
    """
    views, channels, height, width = features.shape
    if (
        not features.is_contiguous(memory_format=torch.channels_last)
        or channels % 4
        or grid.dtype != torch.float32
        or grid.shape != (views, volume.shape[1], height * width, 2)
        or volume.dtype != torch.float32
        or volume.shape != (views, grid.shape[1], height, width)
        or not 1 <= offset < views
    ):
        msg = "Expected channels-last features (C % 4 == 0), an fp32 VD(HW)2 grid and a compatible fp32 cost volume"
        raise ValueError(msg)
    lib.kernel("plane_sweep", features.dtype)(
        features.permute(0, 2, 3, 1),  # NHWC memory: a contiguous view
        grid.contiguous(),
        volume,
        views,
        grid.shape[1],
        height,
        width,
        channels,
        offset,
        threads=volume.numel() * 32,
        group_size=256,
    )
