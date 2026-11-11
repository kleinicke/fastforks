"""Metal kernels for optical flow on Apple MPS (`flow_kernels.metal`): RAFT's correlation lookup without a volume."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks._core.metal import MetalLibrary

load = metal_library("flow_metal", Path(__file__).with_name("flow_kernels.metal"))

GROUP = 128  # fixed: threads per group (the kernel's SIMDS = 4 SIMD groups, one per pixel and level)
CHANNELS = (128, 256, 384, 512)  # fixed: the kernel's instantiations (`corr_window_c<C>`)
MAX_RADIUS = 4  # fixed: the kernel's MAX_SIDE = 2 * 4 + 2


class CorrWindows:
    """RAFT's correlation pyramid and window lookup with the volume never built (same values up to rounding).

    Level l of the pyramid is `f1 . pool^l(f2) / sqrt(C)`: only `f1` and the pooled `f2` levels are kept (O(HW)
    memory instead of O((HW)^2)); `corr_window` takes each window's dot products on the fly.
    """

    def __init__(self, lib: MetalLibrary, fmap1: torch.Tensor, levels: Sequence[torch.Tensor], radius: int) -> None:
        """`fmap1` `[B, C, h, w]`, `levels`: `fmap2` and its pooled levels `[B, C, h_l, w_l]`, all one dtype.

        Raises:
            ValueError: Shapes, channels (128, 256, 384 or 512) or a radius the kernel cannot take.
        """
        b, c, h, w = fmap1.shape
        if c not in CHANNELS or not 0 <= radius <= MAX_RADIUS:
            msg = f"corr_window: {c} channels (one of {CHANNELS}), radius {radius} (at most {MAX_RADIUS})"
            raise ValueError(msg)
        if any(f.shape[:2] != (b, c) or f.dtype != fmap1.dtype for f in levels):
            msg = f"corr_window: levels {[tuple(f.shape) for f in levels]} do not match fmap1 {tuple(fmap1.shape)}"
            raise ValueError(msg)
        self.lib, self.radius, self.shape = lib, radius, (b, h, w)
        self.f1 = fmap1.permute(0, 2, 3, 1).contiguous()
        self.f2 = torch.cat([f.permute(0, 2, 3, 1).reshape(-1) for f in levels])
        meta, o = [], 0
        for f in levels:
            meta += [o, f.shape[2], f.shape[3]]
            o += f.numel()
        if o >= 2**32:
            msg = f"corr_window: {o} pyramid elements overflow the kernel's 32-bit offsets"
            raise ValueError(msg)
        self.meta = torch.tensor(meta, dtype=torch.int32, device=fmap1.device)
        self.levels, self.scale = len(levels), 1.0 / math.sqrt(c)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """The interpolated correlations `[B, levels * (2r + 1)^2, h, w]` around `coords` `[B, 2, h, w]` (x, y)."""
        b, h, w = self.shape
        if coords.shape != (b, 2, h, w):
            msg = f"corr_window: coords {tuple(coords.shape)} for a {b} x {h} x {w} feature map"
            raise ValueError(msg)
        k = 2 * self.radius + 1
        out = self.f1.new_empty(b, self.levels * k * k, h, w)
        p = h * w
        self.lib.kernel(f"corr_window_c{self.f1.shape[-1]}", self.f1.dtype)(
            out, self.f1, self.f2, coords.float().contiguous(), self.meta, p, self.levels, self.radius, self.scale,
            threads=b * p * self.levels * 32, group_size=GROUP,
        )  # fmt: skip
        return out
