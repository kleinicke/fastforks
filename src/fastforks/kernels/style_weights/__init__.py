"""StyleGAN2 filter modulation, demodulation and fp16 channels-last packing."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"style_weights": "pppiiiff"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def pack(weight: torch.Tensor, style: torch.Tensor, gain: float, epsilon: float, *, demodulate: bool) -> torch.Tensor:
    """Scale a single face's fp16 OIHW filter in fp32 and store the result directly in cuDNN's channels-last layout."""
    outputs, channels, kh, kw = weight.shape
    if (
        not weight.is_contiguous(memory_format=torch.channels_last)
        or weight.dtype != torch.float16
        or style.dtype != weight.dtype
        or style.device != weight.device
        or style.shape != (1, channels)
        or not style.is_contiguous()
    ):
        msg = "Expected channels-last fp16 filters and a contiguous per-input-channel style for one face"
        raise ValueError(msg)
    out = torch.empty_like(weight, memory_format=torch.channels_last)
    load(weight.device).launch(
        "style_weights",
        (outputs, 1, 1),
        (256, 1, 1),
        weight,
        style,
        out,
        channels,
        channels * kh * kw,
        int(demodulate),
        epsilon,
        gain,
    )
    return out
