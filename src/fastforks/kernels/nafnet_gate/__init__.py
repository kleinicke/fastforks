"""NAFNet's depthwise 3 x 3 convolution and SimpleGate in one CUDA launch."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"depthwise_gate": "ppppiiil", "depthwise_gate_tile": "ppppiiil", "channel_norm": "ppilf"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


NORM_CHANNELS = (64, 128, 256, 512, 1024)
_NORM_ROWS = 8  # one warp per pixel
_TILE_CHANNELS = 64  # 32 lanes each load two channels
_TILE_WIDTH = 16  # four warps each compute four neighboring pixels
_MAX_GRID_X, _MAX_GRID_Y = 2**31 - 1, 65535
_HALF2_ALIGNMENT = 4


def depthwise_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Gate an NCHW tensor using kernel-major weights [9, 2C] and bias [2C]."""
    n, twice, h, w = x.shape
    c = twice // 2
    if twice % 2 or weight.shape != (9, twice) or bias.shape != (twice,):
        msg = "Expected an even input width, [9, 2C] weights and [2C] biases"
        raise ValueError(msg)
    if any(v.dtype != torch.float16 or v.device != x.device for v in (x, weight, bias)):
        msg = "Expected fp16 activations and weights on the same device"
        raise ValueError(msg)
    out = torch.empty((n, c, h, w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    packed = x.contiguous(memory_format=torch.channels_last)
    weight, bias = weight.contiguous(), bias.contiguous()
    tiles = n * h * ((w + _TILE_WIDTH - 1) // _TILE_WIDTH)
    groups = c // _TILE_CHANNELS
    library = load(x.device)
    # A contiguous view can start on an odd half. The scalar kernel accepts that alignment.
    if (
        0 < groups <= _MAX_GRID_Y
        and c % _TILE_CHANNELS == 0
        and 0 < tiles <= _MAX_GRID_X
        and all(t.data_ptr() % _HALF2_ALIGNMENT == 0 for t in (packed, weight, bias))
    ):
        library.launch(
            "depthwise_gate_tile", (tiles, groups, 1), (32, 4, 1), packed, weight, bias, out, h, w, c // 2, n * h
        )
        return out
    library.launch(
        "depthwise_gate",
        *linear(out.numel()),
        packed,
        weight,
        bias,
        out,
        h,
        w,
        c,
        out.numel(),
    )
    return out


def channel_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize each pixel over its channels without allocating tensors of means and variances."""
    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 4 or x.shape[1] not in NORM_CHANNELS:  # noqa: PLR2004 - NCHW
        msg = f"Expected a CUDA fp16 NCHW tensor with one of {NORM_CHANNELS} channels"
        raise ValueError(msg)
    packed = x.contiguous(memory_format=torch.channels_last)
    if packed.data_ptr() % _HALF2_ALIGNMENT:
        packed = packed.clone()
    n, c, h, w = packed.shape
    rows = n * h * w
    out = torch.empty_like(packed)
    if rows:
        grid = (min(_MAX_GRID_X, (rows + _NORM_ROWS - 1) // _NORM_ROWS), 1, 1)
        load(x.device).launch("channel_norm", grid, (32, _NORM_ROWS, 1), packed, out, c, rows, eps)
    return out
