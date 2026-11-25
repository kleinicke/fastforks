"""Weighted left and warped right features written directly in the layout of Fast-ACVNet's 3-D convolutions."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"weighted": "ppppiiiii"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def weighted(features: torch.Tensor, indices: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """Build the half-precision [B,2C,D,H,W] cost volume from paired features and sorted disparity indices."""
    b, _, depth, h, w = indices.shape
    channels = features.shape[1]
    if (
        min(h, w) < 2  # noqa: PLR2004 - align-corners grid normalization
        or indices.shape[1] != 1
        or features.shape != (2 * b, channels, h, w)
        or attention.shape != indices.shape
        or features.dtype != torch.float16
        or attention.dtype != torch.float32
        or indices.dtype != torch.int64
        or not (features.device == attention.device == indices.device)
    ):
        msg = "Expected paired fp16 BCHW features, int64 B1DHW indices and matching fp32 attention"
        raise ValueError(msg)
    out = torch.empty(
        (b, 2 * channels, depth, h, w),
        device=features.device,
        dtype=features.dtype,
        memory_format=torch.channels_last_3d,
    )
    if out.numel():
        load(features.device).launch(
            "weighted",
            ((w + 7) // 8, min(h, 65535), min(b * depth, 65535)),
            (256, 1, 1),
            features.contiguous(memory_format=torch.channels_last),
            indices.contiguous(),
            attention.contiguous(),
            out,
            b,
            depth,
            h,
            w,
            channels,
        )
    return out
