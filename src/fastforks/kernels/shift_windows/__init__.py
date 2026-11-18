"""Partition shifted Swin windows and put them back without an intermediate roll."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = dict.fromkeys(("window1", "window2", "window8"), "ppiiiiiil")
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def rearrange(
    x: torch.Tensor, size: tuple[int, int], window: int, shift: int, *, reverse: bool = False
) -> torch.Tensor:
    """BNC tokens to [B * windows, window², C], or the inverse when reverse=True."""
    h, w = size
    if (
        x.ndim != 3  # noqa: PLR2004 - BNC tokens
        or x.dtype != torch.float16
        or min(h, w, window) <= 0
        or h % window
        or w % window
        or not 0 <= shift < window
    ):
        msg = "Expected fp16 tokens covering whole square windows and a valid cyclic shift"
        raise ValueError(msg)
    if x.shape[1] != (window**2 if reverse else h * w) or (x.shape[0] * x.shape[1]) % (h * w):
        msg = "Token count does not agree with the spatial grid"
        raise ValueError(msg)
    x = x.contiguous()
    c = x.shape[-1]
    out = x.new_empty(
        (x.shape[0] * x.shape[1] // (h * w), h * w, c) if reverse else (x.shape[0] * h * w // window**2, window**2, c)
    )
    pack = next(v for v in (8, 2, 1) if c % v == 0 and x.data_ptr() % (v * x.element_size()) == 0)
    count = out.numel() // pack
    load(x.device).launch(f"window{pack}", *linear(count), x, out, h, w, c // pack, window, shift, int(reverse), count)
    return out
