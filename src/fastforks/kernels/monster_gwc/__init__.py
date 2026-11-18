"""Selective stereo correlation directly from the two fp16 feature maps."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"selective": "ppppiiiiilllllllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def volume(left: torch.Tensor, right: torch.Tensor, samples: torch.Tensor, depth: int, groups: int) -> torch.Tensor:
    """Warp and correlate one group at a time, converting the final FP32 mean to half."""
    b, channels, h, w = left.shape
    if (
        min(h, w) <= 1
        or groups < 1
        or channels % groups
        or left.dtype != torch.float16
        or right.shape != left.shape
        or right.dtype != left.dtype
        or right.device != left.device
        or samples.shape != (b, depth, h, w)
        or samples.dtype != torch.float32
        or samples.device != left.device
    ):
        msg = "Expected paired fp16 BCHW features and matching float32 BDHW disparities"
        raise ValueError(msg)
    out = left.new_empty((b, groups, depth, h, w))
    if out.numel():
        load(left.device).launch(
            "selective",
            *linear(out.numel() * 32),
            left,
            right,
            samples.contiguous(),
            out,
            groups,
            channels,
            depth,
            h,
            w,
            *left.stride(),
            *right.stride(),
            out.numel(),
        )
    return out
