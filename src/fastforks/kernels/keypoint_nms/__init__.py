"""Threshold and local maximum selection for XFeat detection."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"maxima": "ppiiifl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def maxima(image: torch.Tensor, threshold: float, radius: int) -> torch.Tensor:
    """Return the boolean mask of all local maxima strictly above the threshold."""
    if image.dtype != torch.float32:
        msg = "Expected a float32 score map"
        raise ValueError(msg)
    x = image.contiguous()
    out = torch.empty_like(x, dtype=torch.bool)
    load(x.device).launch("maxima", *linear(x.numel()), x, out, *x.shape[-2:], radius, threshold, x.numel())
    return out
