"""Metal kernels for vision transformers on Apple MPS (`vit_kernels.metal`): Perception Encoder's interleaved RoPE."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

load = metal_library("vit_metal", Path(__file__).with_name("vit_kernels.metal"))


def pe_rope(lib: MetalLibrary, qkv: torch.Tensor, rope: torch.Tensor, npt: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k `[B, H, N, D]` (contiguous) of `qkv` `[B, N, 3, H, D]`: timm's interleaved RoPE after `npt` prefix tokens.

    `rope` `[N - npt, 2 D]` float32 (sin | cos): the math of `fastforks._optimize.transformer._pe_rope` in one pass.
    """
    bsz, n, three, h, d = qkv.shape
    if three != 3 or d % 2 or rope.dtype != torch.float32 or rope.shape != (n - npt, 2 * d):  # noqa: PLR2004 - q, k, v
        msg = f"pe_rope: qkv {tuple(qkv.shape)} [B, N, 3, H, D] (D even), rope {rope.dtype} {tuple(rope.shape)}, {npt=}"
        raise ValueError(msg)
    out = qkv.new_empty(2, bsz, h, n, d)
    lib.kernel("pe_rope", qkv.dtype)(
        out, qkv.contiguous(), rope.contiguous(), bsz, n, h, d, npt, threads=bsz * h * n * d, group_size=256
    )
    return out[0], out[1]
