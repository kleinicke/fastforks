"""Ordered linear splats into a single GIMM-VFI latent buffer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.kernels import cubin_library, linear

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

_CHANNELS = 16
_DTYPES = {torch.float16: 0, torch.float32: 1, torch.bfloat16: 2}
SIGNATURES = {"splat_linear_slice_c16": "pppppiillllllllllllii", "splat_normalize_slice_c16": "ppl"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def _fits(latent: torch.Tensor, flow: torch.Tensor, metric: torch.Tensor) -> bool:
    if latent.ndim != 4:  # noqa: PLR2004 - NCHW
        return False
    b, c, h, w = latent.shape
    return (
        c == _CHANNELS
        and b >= 1
        and 0 < h * w <= 2**24
        and latent.dtype in _DTYPES
        and latent.is_cuda
        and flow.shape == (b, 2, h, w)
        and metric.shape == (b, 1, h, w)
        and flow.dtype in _DTYPES
        and metric.dtype == torch.float32
        and latent.device == flow.device == metric.device
        and all(t.layout == torch.strided and not t.requires_grad for t in (latent, flow, metric))
        and not (latent.is_neg() or metric.is_neg())
    )


def _splat_into(  # noqa: PLR0913 - the kernel inputs and output workspace
    library: CubinModule,
    latent: torch.Tensor,
    flow: torch.Tensor,
    metric: torch.Tensor,
    *,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> None:
    b, _, h, w = latent.shape
    pixels = b * h * w
    elements = pixels * _CHANNELS
    strides = (*latent.stride(), *flow.stride(), metric.stride(0), metric.stride(2), metric.stride(3))
    launch = linear(elements)
    for corner in range(4):
        library.launch(
            "splat_linear_slice_c16", *launch, latent, flow, metric, numerator, denominator,
            h, w, pixels, *strides, _DTYPES[latent.dtype], corner,
        )  # fmt: skip
    library.launch("splat_normalize_slice_c16", *launch, numerator, denominator, elements)


def pair(
    latents: tuple[torch.Tensor, torch.Tensor],
    flows: tuple[torch.Tensor, torch.Tensor],
    metrics: tuple[torch.Tensor, torch.Tensor],
    time: torch.Tensor,
) -> torch.Tensor:
    """Return two normalized 16-channel linear splats in one channels-last FP32 tensor.

    Opposite flow directions are scaled by time and one minus time. Each branch accumulates
    its four bilinear corners in order, sums colliding pixels in FP32, and treats zero total
    weight as a divisor of one. The caller handles deterministic execution and autograd.
    """
    if torch.are_deterministic_algorithms_enabled():
        msg = "GIMM-VFI native splats use unordered additions within each corner"
        raise RuntimeError(msg)
    left, right = latents
    if (
        left.shape != right.shape
        or left.device != right.device
        or any(not _fits(*args) for args in zip(latents, flows, metrics, strict=True))
        or time.shape != (left.shape[0], 1, 1, 1)
        or time.dtype != torch.float32
        or time.device != left.device
        or time.layout != torch.strided
        or time.requires_grad
    ):
        msg = "Expected two 16-channel CUDA latents with matching pixel-flow fields, FP32 metrics and times"
        raise ValueError(msg)

    b, _, h, w = left.shape
    out = left.new_zeros((b, h, w, 2 * _CHANNELS), dtype=torch.float32)
    denominator = left.new_zeros((b, h, w), dtype=torch.float32)
    with torch.cuda.device(left.device):
        library = load(left.device)
        _splat_into(library, left, flows[0] * time, metrics[0], numerator=out[..., :_CHANNELS], denominator=denominator)
        denominator.zero_()
        _splat_into(
            library, right, flows[1] * (1 - time), metrics[1], numerator=out[..., _CHANNELS:], denominator=denominator
        )
    return out.permute(0, 3, 1, 2)
