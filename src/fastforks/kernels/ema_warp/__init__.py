"""Bilinear border warps from EMA-VFI's two pixel-flow fields."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

SIGNATURES = {"warp_f16": "ppppiiiillllllll", "warp_f32": "ppppiiiillllllll"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def pair(first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp both frames, accumulating in float32 and storing directly in their native dtype in channels-last order."""
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
    out = torch.empty((n * 2, c, h, w), device=first.device, dtype=first.dtype, memory_format=torch.channels_last)
    symbol = "warp_f16" if first.dtype == torch.float16 else "warp_f32"
    threads = 256
    grid = ((w * c + threads - 1) // threads, min(h, 65535), min(n * 2, 65535))
    load(first.device).launch(
        symbol, grid, (threads, 1, 1), first, second, flow, out, n, c, h, w, *first.stride(), *flow.stride()
    )
    return out[:n], out[n:]
