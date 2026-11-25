"""Keep four-connected image components of at least twelve pixels."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"keep_twelve": "ppiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def keep_twelve(mask: torch.Tensor) -> torch.Tensor:
    """Apply the AnyLine foreground size threshold to a CUDA bool image batch."""
    _, height, width = mask.shape
    if mask.dtype != torch.bool or height * width >= 2**31:
        msg = "Expected boolean mask planes with fewer than 2**31 pixels"
        raise ValueError(msg)
    out = torch.empty_like(mask, memory_format=torch.contiguous_format)
    load(mask.device).launch("keep_twelve", *linear(mask.numel()), mask.contiguous(), out, height, width, mask.numel())
    return out
