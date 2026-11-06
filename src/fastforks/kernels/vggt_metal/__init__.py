"""Metal kernels for VGGT's aggregator on Apple MPS (`rope_kernels.metal`)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    import torch

    from fastforks._core.metal import MetalLibrary

load = metal_library("vggt_metal", Path(__file__).with_name("rope_kernels.metal"))

HEAD_DIM = 64  # the only head size the kernel takes


def qk_norm_rope(  # noqa: PLR0913, PLR0917 - the kernel's operands
    lib: MetalLibrary,
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    affine: tuple[torch.Tensor, torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm of q and k with `affine` (`[2, 64]` weights, biases: q, k), then RoPE with the per-token tables.

    `qk` `[B, N, 2, H, 64]` (contiguous), `cos` / `sin` `[B * N, 64]` (sin sign-folded) -> q, k `[B, H, N, 64]`.
    """
    bsz, n, _, h, d = qk.shape
    if d != HEAD_DIM:
        msg = f"qk_norm_rope takes head size {HEAD_DIM}, got {d}"
        raise ValueError(msg)
    out = qk.new_empty(2, bsz, h, n, d)
    w, b = affine
    lib.kernel("qk_norm_rope", qk.dtype)(
        out,
        qk.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        w.contiguous(),
        b.contiguous(),
        bsz * n,
        n,
        h,
        eps,
        threads=bsz * n * 2 * h * 32,
        group_size=256,
    )
    return out[0], out[1]
