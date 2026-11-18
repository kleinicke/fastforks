"""Bilinear decoder upsampling and skip concatenation into one channels-last tensor."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"resize_cat": "pppiiiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def resize_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Return cat([interpolate(x, skip spatial size), skip], dim=1) in fp16 channels-last layout."""
    n, c, h, w = x.shape
    b, k, oh, ow = skip.shape
    if n != b or (c % 2 or k % 2) or x.dtype != torch.float16 or skip.dtype != x.dtype or x.device != skip.device:
        msg = "Expected two fp16 feature maps with equal batch size, even channel widths and a common CUDA device"
        raise ValueError(msg)
    x = x.contiguous(memory_format=torch.channels_last)
    skip = skip.contiguous(memory_format=torch.channels_last)
    if x.data_ptr() % 4 or skip.data_ptr() % 4:
        msg = "Half2 feature pairs must be 4-byte aligned"
        raise ValueError(msg)
    out = torch.empty((n, c + k, oh, ow), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    count = out.numel() // 2
    load(x.device).launch("resize_cat", *linear(count), x, skip, out, h, w, oh, ow, c // 2, k // 2, count)
    return out
