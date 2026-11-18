"""Bilinear border warps from EMA-VFI's two pixel-flow fields."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"warp_f16": "ppppiiiillllllllil", "warp_f32": "ppppiiiillllllllil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def _sample(
    first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor, dest: torch.Tensor, right_offset: int
) -> None:
    n, c, h, w = first.shape
    if (
        min(n, c, h, w) < 1
        or w * c >= 2**32
        or second.shape != first.shape
        or second.stride() != first.stride()
        or first.dtype not in {torch.float16, torch.float32}
        or second.dtype != first.dtype
        or flow.shape != (n, 4, h, w)
        or flow.dtype != torch.float32
        or first.device != second.device
        or first.device != flow.device
    ):
        msg = "Expected matching strided FP16/FP32 frames and an FP32 pair of pixel-flow fields on the same device"
        raise ValueError(msg)
    symbol = "warp_f16" if first.dtype == torch.float16 else "warp_f32"
    threads = 256
    grid = ((w * c + threads - 1) // threads, min(h, 65535), min(n * 2, 65535))
    load(first.device).launch(
        symbol,
        grid,
        (threads, 1, 1),
        first,
        second,
        flow,
        dest,
        n,
        c,
        h,
        w,
        *first.stride(),
        *flow.stride(),
        dest.stride(3),
        right_offset,
    )


def pair(first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp both frames, accumulating in float32 and storing in their native dtype in channels-last order."""
    n, c, h, w = first.shape
    out = torch.empty((n * 2, c, h, w), dtype=first.dtype, device=first.device, memory_format=torch.channels_last)
    _sample(first, second, flow, out, first.numel())
    return out[:n], out[n:]


def stack(inputs: tuple[torch.Tensor, ...], features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Write the U-Net activations and two sampled feature maps directly into the convolution input."""
    n, c, h, w = features.shape
    batch = n // 2
    if n % 2 or not inputs or any(x.shape[0] != batch or x.shape[2:] != (h, w) for x in inputs):
        msg = "Expected a pair of feature batches and matching U-Net activations"
        raise ValueError(msg)
    prefix = sum(x.shape[1] for x in inputs)
    out = torch.empty(
        (batch, prefix + 2 * c, h, w), dtype=features.dtype, device=features.device, memory_format=torch.channels_last
    )
    if len(inputs) == 1:
        out[:, :prefix].copy_(inputs[0])
    else:
        torch.cat(inputs, 1, out=out[:, :prefix])
    _sample(features[:batch], features[batch:], flow, out[:, prefix:], c)
    return out
