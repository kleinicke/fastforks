"""CUDA helpers for streaming groups of shifted windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fastforks._core.kernels import linear
from fastforks.kernels import shift_windows

from . import shift_mask

if TYPE_CHECKING:
    from collections.abc import Callable

QKV_TILE_BYTES = 64 << 20
WINDOW_STREAM_BYTES = 256 << 20


def can_stream(x: torch.Tensor, size: tuple[int, int], window: int) -> bool:
    """The raw kernels require an aligned FP16 raster of complete windows."""
    height, width = size
    return (
        x.is_cuda
        and x.dtype == torch.float16
        and x.shape[1] == height * width
        and x.shape[-1] % 2 == 0
        and x.is_contiguous()
        and x.data_ptr() % 4 == 0
        and height % window == width % window == 0
    )


def raster_regions(
    size: tuple[int, int], window: int, shift: int, x: torch.Tensor
) -> tuple[tuple[tuple[int, int, int, int], ...], torch.Tensor | None]:
    """All shifted windows share one of the four boundary masks."""
    nr, nc = (side // window for side in size)
    if not shift:
        return ((0, 0, nr, nc),), None
    masks = shift_mask(2 * window, 2 * window, window, shift, x.device, dtype=x.dtype)
    return ((0, 0, nr - 1, nc - 1), (0, nc - 1, nr - 1, 1), (nr - 1, 0, 1, nc - 1), (nr - 1, nc - 1, 1, 1)), masks


def window_tiles(  # noqa: PLR0913 - tile geometry and the per-window transformer
    x: torch.Tensor,
    size: tuple[int, int],
    *,
    window: int,
    shift: int,
    qkv_width: int,
    bias: torch.Tensor,
    transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Read groups of windows directly from the raster and scatter transformed tokens back."""
    h, w = size
    c = x.shape[-1]
    regions, masks = raster_regions(size, window, shift, x)
    kernels = shift_windows.load(x.device)
    out = torch.empty_like(x)
    step = max(1, QKV_TILE_BYTES // (window**2 * qkv_width * x.element_size()))
    for kind, (r0, c0, rows, cols) in enumerate(regions):
        tile_bias = bias if masks is None or not kind else bias + masks[kind]
        for b in range(x.shape[0]):
            for first in range(0, rows * cols, step):
                raw = x.new_empty((min(step, rows * cols - first), window**2, c))
                threads = raw.numel() // 2
                geometry = h, w, c // 2, window, shift, r0, c0, cols, first, threads
                kernels.launch("stream_window2", *linear(threads), x[b], None, raw, *geometry)
                result = transform(raw, tile_bias)
                kernels.launch("scatter_windows2", *linear(threads), result, out[b], *geometry)
                del raw, result
    return out
