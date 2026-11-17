"""Fixed-point, separable Pillow resampling on CUDA."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"resample_u8": "ppppiiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def axis(x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, *, vertical: bool) -> torch.Tensor:
    """Apply the shared Pillow integer coefficients along one axis of BCHW uint8."""
    batch, channels, height, width = x.shape
    size, taps = indices.shape
    if x.dtype != torch.uint8 or indices.dtype != torch.int64 or weights.dtype != torch.int32:
        msg = "Expected a uint8 image, int64 source indices and int32 coefficients"
        raise ValueError(msg)
    out = x.new_empty((batch, channels, size if vertical else height, width if vertical else size))
    load(x.device).launch(
        "resample_u8",
        *linear(out.numel()),
        x.contiguous(),
        indices,
        weights,
        out,
        height,
        width,
        size,
        taps,
        int(vertical),
        out.numel(),
    )
    return out
