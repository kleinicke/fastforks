"""Omni-DC's three-scale finite-difference normal operator on the quarter-resolution depth grid."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"pool_edges": "ppiil", "normal": "ppppiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_SCALE = 4


def apply(image: torch.Tensor, diagonal: torch.Tensor) -> torch.Tensor:
    """Apply AᵀCA + diag(d) to the single-channel search direction for each image."""
    batch, channels, height, width = image.shape
    if (
        channels != 1
        or height % _SCALE
        or width % _SCALE
        or image.dtype != torch.float32
        or diagonal.shape != image.shape
        or diagonal.dtype != image.dtype
        or diagonal.device != image.device
    ):
        msg = "Expected matching float32 [B, 1, H, W] grids with spatial dimensions divisible by four"
        raise ValueError(msg)
    image = image.contiguous()
    out = torch.empty_like(image)
    cells = batch * ((height // 2) * (width // 2) + (height // 4) * (width // 4))
    edges = image.new_empty((cells, 2))
    kernel = load(image.device)
    kernel.launch("pool_edges", *linear(cells), image, edges, height, width, cells)
    kernel.launch("normal", *linear(out.numel()), image, diagonal.contiguous(), edges, out, height, width, out.numel())
    return out
