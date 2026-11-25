"""CUDA conversion from RGB image tensors to BGR bytes."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"bgr_bytes": "ppll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def levels(image: torch.Tensor) -> torch.Tensor:
    """Reorder RGB planes and round normalized inputs to eight-bit levels."""
    if image.dtype == torch.uint8:
        return image.flip(1)
    x = image.float().contiguous()
    out = torch.empty_like(x, dtype=torch.uint8)
    load(x.device).launch("bgr_bytes", *linear(x.numel()), x, out, x.shape[2] * x.shape[3], x.numel())
    return out
