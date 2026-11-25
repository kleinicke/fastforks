"""Bilinear border warps from frame interpolation from pixel-flow fields."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"warp_f16": "ppppiiiillllllllil", "warp_f32": "ppppiiiillllllllil"}
SIGNATURES["warp_f16x4"] = SIGNATURES["warp_f16"]
SIGNATURES.update(dict.fromkeys(("warp_write_f16", "warp_write_f32", "warp_write_f16x4"), "ppp" + "i" * 6 + "l" * 11))
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def _packed(*views: torch.Tensor) -> bool:
    return all(
        x.dtype == torch.float16
        and x.stride(1) == 1
        and x.data_ptr() % 8 == 0
        and not any(x.stride(i) % 4 for i in (0, 2, 3))
        for x in views
    )


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
    packed = c % 4 == 0 and right_offset % 4 == 0 and _packed(first, second, dest)
    pack = 4 if packed else 1
    if packed:
        symbol = "warp_f16x4"
    grid = ((w * (c // pack) + threads - 1) // threads, min(h, 65535), min(n * 2, 65535))
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


def batch(first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Both warps as one channels-last batch `[2N, C, H, W]`: `first` at flow channels 0-1, then `second` at 2-3.

    Bilinear, border, align-corners grid: pixel `(x, y)` reads `(x + flow_x, y + flow_y)` clipped to the image,
    accumulated in float32 and stored in the frames' dtype.
    """
    n, c, h, w = first.shape
    out = torch.empty((n * 2, c, h, w), dtype=first.dtype, device=first.device, memory_format=torch.channels_last)
    _sample(first, second, flow, out, first.numel())
    return out


def pair(first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp both frames, accumulating in float32 and storing in their native dtype in channels-last order."""
    out = batch(first, second, flow)
    return out[: first.shape[0]], out[first.shape[0] :]


def write_rows(image: torch.Tensor, flow: torch.Tensor, out: torch.Tensor, y0: int = 0) -> None:
    """Sample a half or float image directly into a channels-last half destination, starting at row y0."""
    n, c, h, w = image.shape
    rows = out.shape[-2]
    if (
        min(n, c, h, w) < 1
        or w * c >= 2**32
        or not image.is_cuda
        or flow.dtype != torch.float32
        or out.dtype != torch.float16
        or image.dtype not in {torch.float16, torch.float32}
        or flow.shape != (n, 2, h, w)
        or out.shape != (n, c, rows, w)
        or image.device != flow.device
        or image.device != out.device
        or not (0 <= y0 < y0 + rows <= h)
        or out.stride(1) != 1
    ):
        msg = "Expected a CUDA pixel-flow field and matching FP16 output rows in channels-last layout"
        raise ValueError(msg)
    symbol = "warp_write_f16" if image.dtype == torch.float16 else "warp_write_f32"
    packed = c % 4 == 0 and _packed(image, out)
    pack = 4 if packed else 1
    if packed:
        symbol = "warp_write_f16x4"
    load(image.device).launch(
        symbol,
        ((w * (c // pack) + 255) // 256, min(rows, 65535), min(n, 65535)),
        (256, 1, 1),
        image,
        flow,
        out,
        n,
        c,
        h,
        w,
        y0,
        rows,
        *image.stride(),
        *flow.stride(),
        out.stride(0),
        out.stride(2),
        out.stride(3),
    )


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
    else:  # cat then one strided copy: `cat(..., out=out[:, :prefix])` is 2.5x slower into the channel slice
        out[:, :prefix].copy_(torch.cat([x.to(out.dtype) for x in inputs], 1))
    _sample(features[:batch], features[batch:], flow, out[:, prefix:], c)
    return out
