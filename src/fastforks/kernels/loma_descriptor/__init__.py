"""LoMa sparse feature fusion at the four neighbors of each detected keypoint."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"descriptors": "ppppppiiiifflllllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def blend(  # noqa: PLR0913, PLR0917 - the last two scales and the interpolation stencil
    prior: torch.Tensor,
    delta: torch.Tensor,
    ix: torch.Tensor,
    iy: torch.Tensor,
    weights: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Interpolate the two fp16 scales into one fp32 descriptor per keypoint."""
    b, c, ph, pw = prior.shape
    _, n = ix.shape
    h, w = size
    if b != 1 or delta.shape != (c, 4, n) or any(t.shape != (4, n) for t in (ix, iy, weights)):
        msg = "Expected one prior map and four projected neighbors per keypoint"
        raise ValueError(msg)
    if (prior.dtype, delta.dtype, ix.dtype, iy.dtype, weights.dtype) != (
        torch.float16,
        torch.float16,
        torch.int64,
        torch.int64,
        torch.float32,
    ):
        msg = "Expected fp16 decoder features, int64 indices and fp32 interpolation weights"
        raise ValueError(msg)
    if any(t.device != prior.device for t in (delta, ix, iy, weights)):
        msg = "All decoder tensors must be on the same device"
        raise ValueError(msg)
    out = prior.new_empty((n, c), dtype=torch.float32)
    load(prior.device).launch(
        "descriptors",
        *linear(out.numel()),
        prior,
        delta,
        ix.contiguous(),
        iy.contiguous(),
        weights.contiguous(),
        out,
        n,
        c,
        ph,
        pw,
        ph / h,
        pw / w,
        *prior.stride()[1:],
        *delta.stride(),
        out.numel(),
    )
    return out
