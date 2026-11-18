"""ReCoVEr's constant correlation response and flow channels in a single output."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

_MOTION_CHANNELS = 126
_TILE = 5

SIGNATURES = {"pack": "ppppiilllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def pack(motion: torch.Tensor, tile: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Add the 5x5 bias tile, rectify and append the flow field as channels-last fp16."""
    b, channels, h, w = motion.shape
    if (
        channels != _MOTION_CHANNELS
        or min(h, w) < _TILE
        or tile.shape != (1, channels, _TILE, _TILE)
        or flow.shape != (b, 2, h, w)
        or any(t.dtype != torch.float16 or t.device != motion.device for t in (motion, tile, flow))
    ):
        msg = "Expected 126 motion channels, a 5x5 bias tile and a two-channel fp16 flow"
        raise ValueError(msg)
    out = torch.empty((b, 128, h, w), device=motion.device, dtype=motion.dtype, memory_format=torch.channels_last)
    cl = torch.channels_last
    load(motion.device).launch(
        "pack",
        *linear(out.numel()),
        motion.contiguous(memory_format=cl),
        tile.contiguous(memory_format=cl),
        flow,
        out,
        h,
        w,
        *flow.stride(),
        out.numel(),
    )
    return out
