"""Fuse DAT's two gated features directly into the projection's dense BNC input."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"interaction": "pppppiilllllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def interaction(
    channel_features: torch.Tensor,
    spatial_features: torch.Tensor,
    channel_logits: torch.Tensor,
    spatial_logits: torch.Tensor,
) -> torch.Tensor:
    """Combine BNC features under per-channel and per-pixel sigmoid gates."""
    b, n, c = channel_features.shape
    if spatial_features.shape != (b, n, c) or channel_logits.shape != (b, 1, c) or spatial_logits.shape != (b, n, 1):
        msg = "Expected BNC features with B1C channel gates and BN1 spatial gates"
        raise ValueError(msg)
    tensors = (channel_features, spatial_features, channel_logits, spatial_logits)
    if any(t.dtype != torch.float16 or t.device != channel_features.device for t in tensors):
        msg = "DAT interactions require fp16 tensors on the same device"
        raise ValueError(msg)
    out = channel_features.new_empty((b, n, c))
    load(out.device).launch(
        "interaction",
        *linear(out.numel()),
        channel_features,
        spatial_features,
        channel_logits.sigmoid().contiguous(),
        spatial_logits.sigmoid().contiguous(),
        out,
        n,
        c,
        *channel_features.stride(),
        *spatial_features.stride(),
        out.numel(),
    )
    return out
