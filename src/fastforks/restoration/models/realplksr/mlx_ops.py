"""MLX lowerings of RealPLKSR's export ops: the partial large-kernel convolution (`model.plk_op`) and Mish.

Imported by the MLX port's runner: registers the ops with `fastforks.shared.mlx_graph`. Mish is one Metal kernel
(`fastforks.kernels.graph_mlx.ops.mish`).

`mx.conv2d` runs the 17 x 17 conv of 16 channels at about 0.7 T multiply-adds per second (M1 Pro, 512 x 512: 26.3 ms
float16, 31.2 ms float32; 767 of the network's 1958 ms), and a direct Metal kernel is no faster (`conv_mlx.direct`:
26.5 ms at its best block size). `mx.matmul` reaches 1.8 T: in channels-last memory the `KW` pixels of a kernel row
are `KW * C` consecutive values, so every pixel's row window is a strided view of the padded input, and one GEMM of
these windows with the `[KW * C, KH * C]` weight gives each input row's product with every kernel row. The output is
the sum of `KH` of them along the diagonal (row `y + r`, kernel row `r`), again a strided view: 15.2 ms float16,
20.5 ms float32 (16.8 ms inside the network with the channel split and concatenation).
"""

from __future__ import annotations

import torch

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared.mlx_graph import CL_OPS, OPS

from . import model  # noqa: F401 - registers the ops

# The row products are `KH` times the output (147 MB at 512 x 512 in float16, and as much for the GEMM's contiguous
# windows): computed for bands of output rows of this size, 3 at 512 x 512, at the same speed and 220 MB less peak
BAND_BYTES = 64 << 20


def _rows_band(xp: mx.array, weight: mx.array, h: int, kh: int, co: int) -> mx.array:
    """Output rows of the padded band `xp` [B, h + KH - 1, WP, C] (`weight`: `[KW * C, KH * CO]`)."""
    b, hp, wp, c = xp.shape
    kw = weight.shape[0] // c
    wd = wp - kw + 1
    windows = mx.as_strided(xp, (b, hp, wd, kw * c), (hp * wp * c, wp * c, c, 1))
    z = windows @ weight  # [B, h + KH - 1, W, KH * CO]: every input row against every kernel row
    row = wd * kh * co
    diagonal = mx.as_strided(z, (b, h, wd, kh, co), (hp * row, row, kh * co, row + co, 1))
    return mx.sum(diagonal, axis=3)


def rows_conv(x: mx.array, w: mx.array) -> mx.array:
    """Stride-1 convolution of NHWC `x` with the OIHW kernel `w` (odd sizes, zero padding `k // 2`), as row GEMMs."""
    b, h, wd, _ = x.shape
    co, _, kh, kw = w.shape
    ph, pw = kh // 2, kw // 2
    xp = mx.pad(x, [(0, 0), (ph, ph), (pw, pw), (0, 0)])
    weight = mx.transpose(w, (3, 1, 2, 0)).reshape(-1, kh * co)  # [(dx, ci), (row, co)]
    rows = max(1, BAND_BYTES // (b * wd * kh * co * x.dtype.size) - (kh - 1))
    tops = range(0, h, rows)
    bands = [_rows_band(xp[:, y : min(y + rows, h) + kh - 1], weight, min(rows, h - y), kh, co) for y in tops]
    return bands[0] if len(bands) == 1 else mx.concatenate(bands, axis=1)


def _plk(x: mx.array, w: mx.array, bias: mx.array | None) -> mx.array:
    """NCHW views in and out; the data is channels-last."""
    xl = mx.transpose(x, (0, 2, 3, 1))
    c = w.shape[0]
    y = rows_conv(xl[..., :c], w.astype(x.dtype))
    if bias is not None:
        y = y + bias.astype(x.dtype)
    return mx.transpose(mx.concatenate([y, xl[..., c:]], axis=-1), (0, 3, 1, 2))


OPS[torch.ops.fastforks.realplksr_plk.default] = CL_OPS[torch.ops.fastforks.realplksr_plk.default] = _plk


def _mish(x: mx.array) -> mx.array:
    """Elementwise: on the channels-last data of a 4-D view (contiguous there), so the kernel copies nothing."""
    if x.ndim != 4:  # noqa: PLR2004
        return ops.mish(x)
    return mx.transpose(ops.mish(mx.transpose(x, (0, 2, 3, 1))), (0, 3, 1, 2))


OPS[torch.ops.fastforks.realplksr_mish.default] = CL_OPS[torch.ops.fastforks.realplksr_mish.default] = _mish
