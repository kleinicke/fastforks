"""DepthSplat cost-volume reduction from mixed precision features and projected coordinates."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"plane_float": "ppppiiiiiil", "plane_half": "ppppiiiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def accumulate(  # noqa: PLR0913, PLR0917 - cost volume and neighbour index
    ref: torch.Tensor, target: torch.Tensor, grid: torch.Tensor, volume: torch.Tensor, neighbor: int, neighbors: int
) -> None:
    """Accumulate one neighbour's bilinear correlations and average after the final neighbour."""
    b, c, h, w = ref.shape
    if (
        target.shape != ref.shape
        or target.device != ref.device
        or target.dtype != ref.dtype
        or ref.dtype not in (torch.float16, torch.float32)
        or not ref.is_contiguous(memory_format=torch.channels_last)
        or not target.is_contiguous(memory_format=torch.channels_last)
        or grid.shape != (b, volume.shape[1], h * w, 2)
        or grid.dtype != torch.float16
        or grid.device != ref.device
        or not grid.is_contiguous()
        or volume.shape != (b, grid.shape[1], h, w)
        or volume.dtype != torch.float32
        or volume.device != ref.device
        or not volume.is_contiguous()
        or not 0 <= neighbor < neighbors
    ):
        msg = "Expected matching channels-last feature pairs, contiguous fp16 grid and fp32 cost volume"
        raise ValueError(msg)
    name = "plane_float" if ref.dtype == torch.float32 else "plane_half"
    load(ref.device).launch(
        name,
        *linear(volume.numel() * 32),
        ref,
        target,
        grid,
        volume,
        c,
        h,
        w,
        grid.shape[1],
        neighbor,
        neighbors,
        volume.numel(),
    )
