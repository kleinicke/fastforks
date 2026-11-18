"""RF-DETR's bilinear square resize and RGB normalization in one pass."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"resize_normalize": "ppppiiilllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def prepare(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, side: int) -> torch.Tensor:
    """Read arbitrary BCHW float32 strides and write the normalized fp16 network input."""
    b, channels, height, width = image.shape
    if (
        channels != 3  # noqa: PLR2004 - RGB
        or min(b, height, width, side) < 1
        or image.dtype != torch.float32
        or any(t.device != image.device or t.dtype != torch.float32 or t.numel() != channels for t in (mean, std))
    ):
        msg = "Expected float32 RGB and per-channel normalization on the same CUDA device"
        raise ValueError(msg)
    out = torch.empty((b, channels, side, side), device=image.device, dtype=torch.float16)
    load(image.device).launch(
        "resize_normalize",
        *linear(out.numel()),
        image,
        mean.contiguous(),
        std.contiguous(),
        out,
        height,
        width,
        side,
        *image.stride(),
        out.numel(),
    )
    return out
