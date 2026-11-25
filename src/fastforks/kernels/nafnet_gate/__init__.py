"""NAFNet's depthwise 3 x 3 convolution and SimpleGate in one CUDA launch."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {
    "depthwise_gate": "ppppiiil",
    "depthwise_gate_tile": "ppppiiil",
    "depthwise_gate_rows": "ppppiiiiiiil",
    "depthwise_gate_rows_tile": "ppppiiiiiiil",
    "channel_norm": "ppilf",
    "scale_replicate": "ppiiiiil",
}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


NORM_CHANNELS = (64, 128, 256, 512, 1024)
_NORM_ROWS = 8  # one warp per pixel
_TILE_CHANNELS = 64  # 32 lanes each load two channels
_TILE_WIDTH = 16  # four warps each compute four neighboring pixels
_MAX_GRID_X, _MAX_GRID_Y = 2**31 - 1, 65535
_HALF2_ALIGNMENT = 4


def _validate_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    twice = x.shape[1]
    if twice % 2 or weight.shape != (9, twice) or bias.shape != (twice,):
        msg = "Expected an even input width, [9, 2C] weights and [2C] biases"
        raise ValueError(msg)
    if any(v.dtype != torch.float16 or v.device != x.device for v in (x, weight, bias)):
        msg = "Expected fp16 activations and weights on the same device"
        raise ValueError(msg)


def _launch_gate(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor, crop: tuple[int, int, int, int] | None
) -> None:
    n, twice, h, w = x.shape
    c = twice // 2
    packed = x.contiguous(memory_format=torch.channels_last)
    weight, bias = weight.contiguous(), bias.contiguous()
    core = h if crop is None else crop[0]
    tiles = n * core * ((w + _TILE_WIDTH - 1) // _TILE_WIDTH)
    groups = c // _TILE_CHANNELS
    library = load(x.device)
    name = "depthwise_gate" if crop is None else "depthwise_gate_rows"
    region = () if crop is None else crop
    # A contiguous view can start on an odd half. The scalar kernel accepts that alignment.
    if (
        0 < groups <= _MAX_GRID_Y
        and c % _TILE_CHANNELS == 0
        and 0 < tiles <= _MAX_GRID_X
        and all(t.data_ptr() % _HALF2_ALIGNMENT == 0 for t in (packed, weight, bias, out))
    ):
        library.launch(
            name + "_tile", (tiles, groups, 1), (32, 4, 1), packed, weight, bias, out, h, w, c // 2, *region, n * core
        )
        return
    count = n * core * w * c
    library.launch(name, *linear(count), packed, weight, bias, out, h, w, c, *region, count)


def depthwise_gate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Gate an NCHW tensor using kernel-major weights [9, 2C] and bias [2C]."""
    _validate_gate(x, weight, bias)
    n, twice, h, w = x.shape
    out = torch.empty((n, twice // 2, h, w), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    _launch_gate(x, weight, bias, out, None)
    return out


def depthwise_gate_rows(  # noqa: PLR0913 - crop geometry alongside the kernel inputs
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    src_top: int,
    dst_top: int,
    rows: int,
) -> None:
    """Gate a row range of ``x`` into the matching range of a separate packed destination."""
    _validate_gate(x, weight, bias)
    n, twice, h, w = x.shape
    if (
        out.ndim != x.ndim
        or (out.shape[0], out.shape[1], out.shape[-1]) != (n, twice // 2, w)
        or not out.is_contiguous(memory_format=torch.channels_last)
        or out.device != x.device
        or out.dtype != x.dtype
        or rows <= 0
        or not 0 <= src_top <= h - rows
        or not 0 <= dst_top <= out.shape[-2] - rows
    ):
        msg = "Expected a matching depthwise gate destination and valid row ranges"
        raise ValueError(msg)
    _launch_gate(x, weight, bias, out, (rows, src_top, out.shape[-2], dst_top))


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


def scale_replicate_(x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Multiply a channels-last feature map by its centered, edge-replicated FP16 scores."""
    n, c, h, w = x.shape
    sn, sc, oh, ow = scores.shape
    if (
        not x.is_cuda
        or x.device != scores.device
        or x.dtype != torch.float16
        or scores.dtype != x.dtype
        or (n, c) != (sn, sc)
        or not 0 < oh <= h
        or not 0 < ow <= w
        or c % 2
        or any(
            not t.is_contiguous(memory_format=torch.channels_last) or t.data_ptr() % _HALF2_ALIGNMENT
            for t in (x, scores)
        )
    ):
        msg = "Expected compatible CUDA channels-last FP16 features and local scores with an even channel width"
        raise ValueError(msg)
    count = x.numel() // 2
    load(x.device).launch("scale_replicate", *linear(count), x, scores, h, w, oh, ow, c // 2, count)
    return x
