"""Fused convex upsampling of a two-channel optical flow field."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"convex": "pppiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def convex(flow: torch.Tensor, mask: torch.Tensor, upsample_factor: int = 8) -> torch.Tensor:
    """Softmax the nine neighbours and interpolate a batch of FP32 flow fields."""
    b, c, h, w = flow.shape
    k = upsample_factor
    if c != 2 or k not in (4, 8) or mask.shape != (b, 9 * k * k, h, w) or flow.dtype != torch.float32:  # noqa: PLR2004 - two flow components
        msg = "Expected a two-channel FP32 flow field and its 4x or 8x convex mask"
        raise ValueError(msg)
    out = flow.new_empty((b, c, h * k, w * k))
    count = b * h * w * k * k
    load(flow.device).launch(
        "convex", *linear(count), flow.contiguous(), mask.float().contiguous(), out, h, w, k, count
    )
    return out
