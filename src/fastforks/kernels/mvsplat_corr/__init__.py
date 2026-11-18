"""MVSplat plane-sweep correlations directly from projected feature coordinates."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"plane_sweep": "pppiiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def accumulate(features: torch.Tensor, grid: torch.Tensor, volume: torch.Tensor, offset: int) -> None:
    """Write the first view offset's cost, then add later offsets to the same FP32 volume."""
    views, channels, height, width = features.shape
    if (
        not features.is_contiguous(memory_format=torch.channels_last)
        or features.dtype != torch.float16
        or not grid.is_contiguous()
        or grid.dtype != torch.float16
        or grid.shape != (views, volume.shape[1], height * width, 2)
        or not volume.is_contiguous()
        or volume.dtype != torch.float32
        or volume.shape != (views, grid.shape[1], height, width)
        or grid.device != features.device
        or volume.device != features.device
        or not 1 <= offset < views
    ):
        msg = "Expected channels-last fp16 features, fp16 VDH(W)2 grid and a compatible fp32 cost volume"
        raise ValueError(msg)
    count = volume.numel()
    load(features.device).launch(
        "plane_sweep", *linear(count * 32), features, grid, volume, views, offset, channels, height, width, count
    )
