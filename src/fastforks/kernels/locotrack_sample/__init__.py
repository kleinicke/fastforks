"""LocoTrack's 7 x 7 local sampling directly into correlation operand layout."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"sample_float": "pppiiiiillllll", "sample_half": "pppiiiiillllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def neighborhood(grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample NHWC frame grids at 256-raster (x, y) track coordinates."""
    b, t, h, w, c = grid.shape
    n = points.shape[1]
    if points.shape != (b, n, t, 2) or points.dtype != torch.float32 or grid.device != points.device:
        msg = "Expected float32 (B, N, T, 2) coordinates on the feature grid device"
        raise ValueError(msg)
    name = {torch.float32: "sample_float", torch.float16: "sample_half"}[grid.dtype]
    out = grid.new_empty((b, n, t, 49, c), dtype=torch.float16)
    load(grid.device).launch(
        name, *linear(out.numel()), grid, points.contiguous(), out, t, h, w, c, n, *grid.stride(), out.numel()
    )
    return out
