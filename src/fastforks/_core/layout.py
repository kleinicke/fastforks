"""Memory-layout conversions faster than torch's strided copy."""

from __future__ import annotations

import torch

NCHW_GEMM_CHANNELS = (8, 512)  # fixed: the identity GEMM's C x C work grows with C; 160 / 320 channels measured


def nchw(x: torch.Tensor) -> torch.Tensor:
    """`x.contiguous()`: a 4-D channels-last tensor as NCHW, the same values.

    torch's strided copy moves a channels-last fp16 / bf16 CUDA map at ~40 GB/s (160 x 512 x 752: 10.9 ms); as the
    identity GEMM `eye(C) @ x^T` it takes 1.8 ms. Each output is one product with 1 plus zeros, so finite values come
    out exactly (a non-finite value would turn the channels of its pixel into NaN).
    """
    lo, hi = NCHW_GEMM_CHANNELS
    if not (
        x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16}
        and x.dim() == 4  # noqa: PLR2004 - BCHW
        and lo <= x.shape[1] <= hi
        and x.shape[1] % 8 == 0
        and not x.is_contiguous()
        and x.is_contiguous(memory_format=torch.channels_last)
        and not (torch.compiler.is_compiling() or torch.compiler.is_exporting())
    ):
        return x.contiguous()
    b, c, h, w = x.shape
    eye = torch.eye(c, device=x.device, dtype=x.dtype)
    return torch.matmul(eye, x.permute(0, 2, 3, 1).reshape(b, h * w, c).transpose(1, 2)).view(b, c, h, w)
