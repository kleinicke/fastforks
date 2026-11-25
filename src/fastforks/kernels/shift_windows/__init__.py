"""Partition shifted Swin windows and put them back without an intermediate roll."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = dict.fromkeys(("window1", "window2", "window8"), "ppiiiiiil")
SIGNATURES.update(dict.fromkeys(("reverse_add2", "reverse_add4"), "ppppiiiiiifi"))
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


def reverse_residual(  # noqa: PLR0913 - three inputs and the window geometry
    windows: torch.Tensor,
    shortcut: torch.Tensor,
    conv: torch.Tensor,
    size: tuple[int, int],
    *,
    window: int,
    shift: int,
    scale: float,
) -> torch.Tensor:
    """Reverse the window partition and add HAT's two residuals with FP16 rounding at each step."""
    h, w = size
    b, hw, c = shortcut.shape
    if (
        min(b, c, h, w, window) < 1
        or hw != h * w
        or h % window
        or w % window
        or not 0 <= shift < window
        or windows.shape != (b * hw // window**2, window**2, c)
        or conv.shape != shortcut.shape
        or not shortcut.is_cuda
        or any(t.device != shortcut.device or t.dtype != torch.float16 for t in (windows, shortcut, conv))
    ):
        msg = "Window and residual tensors must be CUDA FP16 and cover the same grid"
        raise ValueError(msg)
    conv_channels = conv.stride(1)
    if (
        c % 2
        or conv_channels % 2
        or not windows.is_contiguous()
        or not shortcut.is_contiguous()
        or conv.stride(-1) != 1
        or (b > 1 and conv.stride(0) != hw * conv_channels)
        or any(t.data_ptr() % 4 for t in (windows, shortcut, conv))
        or b * hw * max(c, conv_channels) >= 2**31
    ):
        return shortcut + rearrange(windows, size, window, shift, reverse=True) + conv * scale
    pack = 4 if c % 4 == 0 and conv_channels % 4 == 0 else 2
    out = torch.empty_like(shortcut)
    count = out.numel() // pack
    load(windows.device).launch(
        f"reverse_add{pack}", *linear(count), windows, shortcut, conv, out,
        h, w, c // 2, conv_channels // 2, window, shift, scale, count,
    )  # fmt: skip
    return out
