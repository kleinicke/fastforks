"""Metal kernels for sparse 3-D convolution on Apple MPS (`sparse_kernels.metal`)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

load = metal_library("sparse_metal", Path(__file__).with_name("sparse_kernels.metal"))

_TILE = 32  # output rows and channels per threadgroup (and the channel multiple the kernel needs)


class SubmConv:
    """Submanifold 3x3x3 conv (weight `[C_out, 3, 3, 3, C_in]`, FlexGEMM's layout) as one `subm_conv` kernel call."""

    def __init__(self, lib: MetalLibrary, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Lay the weight out as `[27, C_in, C_out]`; channel counts must be multiples of 32."""
        o, c = weight.shape[0], weight.shape[-1]
        if o % _TILE or c % _TILE:
            msg = f"subm_conv needs channel counts that are multiples of {_TILE}, got {c} -> {o}"
            raise ValueError(msg)
        self.lib = lib
        self.weight = weight.detach().reshape(o, 27, c).permute(1, 2, 0).contiguous()
        self.bias = bias.detach()

    def __call__(self, x: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        """`x [N, C_in]`, `nbr [27, N]` (int32 input row per tap, or -1) -> `[N, C_out]`."""
        n, c = x.shape
        o = self.weight.shape[2]
        if x.dtype != self.weight.dtype or self.bias.dtype != x.dtype:
            msg = f"subm_conv: x is {x.dtype}, weight {self.weight.dtype}, bias {self.bias.dtype}; they must match"
            raise TypeError(msg)
        if nbr.dtype != torch.int32 or tuple(nbr.shape) != (27, n):
            msg = f"subm_conv: nbr must be int32 [27, {n}], got {nbr.dtype} {tuple(nbr.shape)}"
            raise ValueError(msg)
        out = x.new_empty(n, o)
        self.lib.kernel("subm_conv", x.dtype)(
            out, x.contiguous(), nbr.contiguous(), self.weight, self.bias, n, c, o,
            threads=-(-n // _TILE) * (o // _TILE) * 128, group_size=128,
        )  # fmt: skip
        return out
