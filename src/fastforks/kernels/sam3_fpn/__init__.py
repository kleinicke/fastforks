"""SAM 3 pixel decoder's nearest upsampling, skip addition and normalization epilogue."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"nearest_add": "pppiiiilllllllll", "nearest_add8": "pppiiil", "affine_relu": "pppiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_PACK = 8  # eight fp16 channels in a 128-bit vector


def nearest_add(coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Add a fine skip map to its doubled nearest-neighbor coarse map."""
    b, c, h, w = coarse.shape
    if skip.shape != (b, c, 2 * h, 2 * w) or coarse.dtype != torch.float16 or skip.dtype != coarse.dtype:
        msg = "Expected fp16 NCHW maps at adjacent 2x pyramid resolutions"
        raise ValueError(msg)
    if skip.device != coarse.device:
        msg = "Pyramid levels must share a device"
        raise ValueError(msg)
    nhwc = coarse.is_contiguous(memory_format=torch.channels_last)
    fmt = torch.channels_last if nhwc else torch.contiguous_format
    out = torch.empty_like(skip, memory_format=fmt)
    k = load(coarse.device)
    if (
        nhwc
        and skip.is_contiguous(memory_format=fmt)
        and c % _PACK == 0
        and not (coarse.data_ptr() | skip.data_ptr()) % 16
    ):
        n = out.numel() // _PACK
        k.launch("nearest_add8", *linear(n), coarse, skip, out, h, w, c // _PACK, n)
    else:
        n = out.numel()
        k.launch("nearest_add", *linear(n), coarse, skip, out, h, w, c, int(nhwc), *coarse.stride(), *skip.stride(), n)
    return out


def affine_relu_(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """In-place GroupNorm affine transform and ReLU of a fresh, dense fp16 convolution result."""
    b, c, h, w = x.shape
    nhwc = x.is_contiguous(memory_format=torch.channels_last)
    if not (nhwc or x.is_contiguous()) or x.dtype != torch.float16:
        msg = "Expected a dense fp16 convolution result"
        raise ValueError(msg)
    if any(t.shape != (b, c) or t.dtype != x.dtype or t.device != x.device for t in (scale, shift)):
        msg = "Expected fp16 affine coefficients [B,C] on the feature device"
        raise ValueError(msg)
    n = x.numel()
    load(x.device).launch("affine_relu", *linear(n), x, scale.contiguous(), shift.contiguous(), c, h * w, int(nhwc), n)
    return x
