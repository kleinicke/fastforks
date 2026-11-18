"""Inference epilogues for dense, compact and parameter-free-attention super-resolution networks."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"bias_activation": "pppiil", "scaled_residual": "ppl", "span_gate": "ppl", "shuffle_residual": "pppiiiil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
_CL = torch.channels_last


def _features(x: torch.Tensor) -> bool:
    return x.ndim == 4 and x.dtype == torch.float16 and x.is_cuda and x.is_contiguous(memory_format=_CL)  # noqa: PLR2004


def bias_activation(x: torch.Tensor, bias: torch.Tensor, prelu: torch.Tensor | None = None) -> torch.Tensor:
    """Add the bias and apply per-channel PReLU or a 0.2 LeakyReLU in place in an NHWC channel slice."""
    n, c, h, w = x.shape
    pitch = x.stride(3)
    if (
        x.dtype != torch.float16
        or not x.is_cuda
        or x.stride(1) != 1
        or pitch < c
        or (h > 1 and x.stride(2) != w * pitch)
        or (n > 1 and x.stride(0) != h * w * pitch)
        or any(
            t.shape != (c,) or t.dtype != x.dtype or t.device != x.device or not t.is_contiguous()
            for t in (bias, prelu)
            if t is not None
        )
    ):
        msg = "Expected fp16 NHWC channel slices and matching contiguous bias/PReLU vectors"
        raise ValueError(msg)
    load(x.device).launch("bias_activation", *linear(x.numel()), x, bias, prelu, c, pitch, x.numel())
    return x


def residual(branch: torch.Tensor, skip: torch.Tensor, *, attention: bool = False) -> torch.Tensor:
    """Write the dense scaled residual or SPAN's gated residual directly into its completed convolution branch."""
    if not (_features(branch) and _features(skip) and branch.shape == skip.shape and branch.device == skip.device):
        msg = "Expected matching channels-last fp16 residual tensors"
        raise ValueError(msg)
    kernel = "span_gate" if attention else "scaled_residual"
    load(branch.device).launch(kernel, *linear(branch.numel()), branch, skip, branch.numel())
    return branch


def shuffle_residual(low: torch.Tensor, skip: torch.Tensor, scale: int) -> torch.Tensor:
    """Pixel-shuffle a low-resolution prediction and add its nearest-neighbour RGB residual in one write."""
    b, c, h, w = skip.shape
    if (
        scale < 1
        or low.shape != (b, c * scale**2, h, w)
        or not (_features(low) and _features(skip))
        or skip.device != low.device
    ):
        msg = "Expected compatible channels-last fp16 packed prediction and low-resolution image"
        raise ValueError(msg)
    out = torch.empty((b, c, h * scale, w * scale), device=low.device, dtype=low.dtype, memory_format=_CL)
    load(low.device).launch("shuffle_residual", *linear(out.numel()), low, skip, out, h, w, c, scale, out.numel())
    return out
