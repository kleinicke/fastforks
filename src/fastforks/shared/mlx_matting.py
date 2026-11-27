"""Pre- and post-processing of the hand-written MLX matting / dichotomous segmentation runners, on the GPU.

What the PyTorch runners do around their networks (`F.interpolate`, `scale`, min-max), in `mx` operations on NCHW
float32 maps, so that an MLX run converts the image once and keeps it on the GPU:

- `resize_aa`: `F.interpolate(mode="bilinear" | "bicubic", antialias=True)` as two matrix products with PyTorch's
  separable antialiasing weights (`mlx_graph._aa_matrix`, the graph translator's);
- `resize`: `F.interpolate(mode="bilinear")` without antialiasing (`graph_mlx.ops.bilinear`);
- `resize_cubic`: `F.interpolate(mode="bicubic", align_corners=False)` without antialiasing, two matrix products;
- `normalise`: the card's `[io]` value range (`fastforks._core.preprocess.scale`).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

from fastforks._core.mlx import mx
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg

if TYPE_CHECKING:
    from fastforks._core.cards import Io

__all__ = ["min_max", "normalise", "resize", "resize_aa", "resize_cubic"]

TINY = 1e-8  # the PyTorch runners' floor of the normalisation divisors (a constant image / map: not 0 / 0)


@functools.lru_cache(maxsize=32)
def _aa(inp: int, out: int, align: bool, cubic: bool) -> mx.array:  # noqa: FBT001 - cache key
    return mx.array(mg._aa_matrix(inp, out, align, None, cubic=cubic))  # noqa: SLF001 - the translator's weights


def resize_aa(x: mx.array, size: tuple[int, int], *, cubic: bool = False, align: bool = False) -> mx.array:
    """Antialiased bilinear (bicubic) resize of NCHW `x` to `size`, as `F.interpolate(..., antialias=True)`."""
    h, w = x.shape[2:]
    if (h, w) == tuple(size):
        return x
    wy, wx = _aa(h, size[0], align, cubic).astype(x.dtype), _aa(w, size[1], align, cubic).astype(x.dtype)
    return mx.matmul(mx.matmul(wy, x), wx.T)


def resize(x: mx.array, size: tuple[int, int], *, align: bool = False) -> mx.array:
    """Bilinear resize of NCHW `x` to `size`, as `F.interpolate(mode="bilinear", align_corners=align)`."""
    h, w = x.shape[2:]
    if (h, w) == tuple(size):
        return x
    if align:
        ratios = ((h - 1) / (size[0] - 1) if size[0] > 1 else 0.0, (w - 1) / (size[1] - 1) if size[1] > 1 else 0.0)
    else:
        ratios = (h / size[0], w / size[1])
    return ops.bilinear(x, size, ratios, align=align)


CUBIC_A = -0.75  # PyTorch's bicubic coefficient


@functools.lru_cache(maxsize=32)
def _cubic(inp: int, out: int) -> mx.array:
    """`[out, inp]` weights of PyTorch's bicubic resize along one axis (`align_corners=False`, edge taps clamped)."""
    src = (np.arange(out) + 0.5) * (inp / out) - 0.5  # not clamped (PyTorch's cubic source index)
    i0 = np.floor(src).astype(int)
    t = src - i0
    a = CUBIC_A

    def near(d: np.ndarray) -> np.ndarray:  # |d| <= 1
        return ((a + 2) * d - (a + 3)) * d * d + 1

    def far(d: np.ndarray) -> np.ndarray:  # 1 < |d| < 2
        return ((a * d - 5 * a) * d + 8 * a) * d - 4 * a

    m = np.zeros((out, inp))
    rows = np.arange(out)
    for k, wk in enumerate((far(t + 1), near(t), near(1 - t), far(2 - t))):
        np.add.at(m, (rows, np.clip(i0 - 1 + k, 0, inp - 1)), wk)
    return mx.array(m.astype(np.float32))


def resize_cubic(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bicubic resize of NCHW float32 `x` to `size`, as `F.interpolate(mode="bicubic", align_corners=False)`."""
    h, w = x.shape[2:]
    if (h, w) == tuple(size):
        return x
    return mx.matmul(mx.matmul(_cubic(h, size[0]), x), _cubic(w, size[1]).T)


def normalise(x: mx.array, io: Io) -> mx.array:
    """`[0, 1]` RGB NCHW -> the value range the network expects (`fastforks._core.preprocess.scale`)."""
    if io.input == "0-255":
        return x * 255.0
    if io.input in {"imagenet", "mean_std"}:
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if io.input == "imagenet" else (io.mean, io.std)
        m = mx.array(mean, dtype=mx.float32).reshape(1, -1, 1, 1)
        s = mx.array(std, dtype=mx.float32).reshape(1, -1, 1, 1)
        return (x - m) / s
    return x


def min_max(y: mx.array, axes: tuple[int, ...]) -> mx.array:
    """`(y - min) / max(max - min, 1e-8)` over `axes` (a constant map stays 0)."""
    lo, hi = y.min(axis=axes, keepdims=True), y.max(axis=axes, keepdims=True)
    return (y - lo) / mx.maximum(hi - lo, TINY)
