"""In-place affine transformation and SiLU with an intermediate FP16 rounding."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"affine_silu_fp16": "pppil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_VECTOR = 8


def apply_(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply per-sample, per-channel affine coefficients and SiLU to channels-last FP16 data."""
    batch, channels, height, width = x.shape
    alignment = _VECTOR * x.element_size()
    if (
        not x.is_cuda
        or x.dtype != torch.float16
        or not x.is_contiguous(memory_format=torch.channels_last)
        or channels % _VECTOR
        or x.data_ptr() % alignment
        or any(
            t.numel() != batch * channels
            or t.dtype != x.dtype
            or t.device != x.device
            or not t.is_contiguous()
            or t.data_ptr() % alignment
            for t in (scale, bias)
        )
    ):
        msg = "Expected aligned channels-last CUDA FP16 data and its contiguous affine coefficients"
        raise ValueError(msg)
    if not x.numel():
        return x
    vectors = channels // _VECTOR
    work = height * width * vectors
    grid, block = linear(work)
    load(x.device).launch("affine_silu_fp16", (grid[0], batch, 1), block, x, scale, bias, vectors, work)
    return x
