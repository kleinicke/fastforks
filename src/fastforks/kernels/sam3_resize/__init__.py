"""Antialiased bilinear RGB resize with SAM 3's input and output level quantization and fp16 normalization."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

_RGB = 3
_SIGNATURE = "ppppiiilllll"
SIGNATURES = {"triangle_tables": "ppiiiii", "resize_rgb_u8": _SIGNATURE, "resize_rgb_f32": _SIGNATURE}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def normalized_rgb(image: torch.Tensor, side: int) -> torch.Tensor:
    """Read byte RGB (or unit-interval floats) at the source resolution and write a normalized fp16 square."""
    b, c, h, w = image.shape
    if c != _RGB or min(b, h, w, side) < 1 or image.dtype not in (torch.uint8, torch.float32):
        msg = "Expected byte or float32 RGB images and a positive output size"
        raise ValueError(msg)
    # One extra coefficient covers rounding at the boundary of a triangle with an integral radius.
    ystride = 2 * max(1, (h + side - 1) // side) + 1
    xstride = 2 * max(1, (w + side - 1) // side) + 1
    bounds = torch.empty((2 * side, 2), device=image.device, dtype=torch.int32)
    weights = torch.empty(side * (ystride + xstride), device=image.device, dtype=torch.float32)
    out = torch.empty((b, c, side, side), device=image.device, dtype=torch.float16)
    lib = load(image.device)
    lib.launch("triangle_tables", *linear(2 * side), bounds, weights, h, w, side, ystride, xstride)
    kernel = "resize_rgb_u8" if image.dtype == torch.uint8 else "resize_rgb_f32"
    lib.launch(
        kernel, *linear(out.numel()), image, bounds, weights, out, side, ystride, xstride, *image.stride(), out.numel()
    )
    return out
