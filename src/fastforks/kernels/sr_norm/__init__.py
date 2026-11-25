"""Vectorized CUDA LayerNorm for restoration token maps."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

TORCH_ORDER_CHANNELS = 240
SIGNATURES = {f"sr_norm{width}_v{vectors}": "pppplifi" for width in (4, 8) for vectors in (1, 2, 4, 8)}
SIGNATURES["sr_norm_torch_240"] = "pppplf"
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def supported(channels: int) -> bool:
    """Whether four or eight half values per thread cover the channel width."""
    return channels > 0 and channels % 4 == 0 and channels <= (2048 if channels % 8 == 0 else 1024)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize a contiguous aligned CUDA fp16 array over its final dimension."""
    channels = x.shape[-1] if x.ndim else 0
    width = 8 if channels % 8 == 0 else 4
    if (
        not supported(channels)
        or not x.is_cuda
        or x.dtype != torch.float16
        or not x.is_contiguous()
        or x.data_ptr() % (width * x.element_size())
        or any(
            t.shape != (channels,) or t.dtype != torch.float32 or t.device != x.device or not t.is_contiguous()
            for t in (weight, bias)
        )
    ):
        msg = "Expected aligned contiguous CUDA fp16 tokens and their same-device fp32 LayerNorm affine parameters"
        raise ValueError(msg)
    rows = x.numel() // channels
    if rows == 0:
        return torch.empty_like(x)
    group = 1
    while group < 32 and group * width < channels:  # noqa: PLR2004 - warp size
        group *= 2
    vectors = next(vectors for vectors in (1, 2, 4, 8) if group * width * vectors >= channels)
    out = torch.empty_like(x)
    load(x.device).launch(
        f"sr_norm{width}_v{vectors}",
        ((rows * group + 255) // 256, 1, 1),
        (256, 1, 1),
        x,
        weight,
        bias,
        out,
        rows,
        channels,
        eps,
        group,
    )
    return out


def layer_norm_240(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply PyTorch's aligned FP16 LayerNorm reduction to prechecked 240-channel rows."""
    rows = x.numel() // TORCH_ORDER_CHANNELS
    out = torch.empty_like(x)
    if rows:
        load(x.device).launch(
            "sr_norm_torch_240", ((rows + 3) // 4, 1, 1), (256, 1, 1), x, weight, bias, out, rows, eps
        )
    return out
