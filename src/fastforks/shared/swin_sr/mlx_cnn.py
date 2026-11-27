"""The 3 x 3 convolution of the hand-written MLX super-resolution CNNs (RRDBNet, SPAN, Compact).

A conv is a dict of arrays (`params`: OHWI weights, bias, and in kernels mode its Winograd-domain weights), applied by
`conv` inside the networks' compiled steps (one compiled function per step kind, the arrays its arguments):

- portable mode: the graph translator's `mlx_graph._conv2d` (its channel padding onto MLX's fast tiles), as the graph
  translation ran these nets; the activation and the residual are `mx.compile`-fused element-wise ops;
- kernels mode: `conv_mlx.conv3x3` (Winograd F(4,3)) where `conv_mlx.supported` takes the shape, its bias, PReLU /
  LeakyReLU and residual fused into the output transform, as the graph translation's kernels mode ran them; other
  shapes (few channels, tiny maps) go to `_conv2d` with the translator's kernels-mode flags (`set_mode`).

Weights scaled by a constant (`params(..., scale=)`: an RDB's `0.2 * conv5(x) + x`) fold the scale in at load, as
the translator's residual fusion does. Their tails run in `mlx_net.bands`' row bands.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from typing import Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_graph as mg
from fastforks.shared.swin_sr import mlx_net as swin

__all__ = ["NONE", "PRELU", "conv", "params", "prelu", "set_mode"]

NONE, PRELU = 0, 2  # `conv_mlx.conv3x3`'s activations (a LeakyReLU is a PReLU of a constant slope)


def _wino(w: Any, m: int, scale: float) -> np.ndarray:
    """`conv_mlx.weights` of a torch OIHW 3 x 3 kernel times `scale` (float64 transform)."""
    from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - Metal only

    ohwi = mx.array(np.ascontiguousarray(w.detach().permute(0, 2, 3, 1).double().numpy() * scale, dtype=np.float32))
    return np.array(conv_ops.weights(ohwi, m, mx.float32))


def params(layer: Any, dtype: mx.Dtype, *, wino: bool, scale: float = 1.0) -> dict[str, mx.array]:
    """An `nn.Conv2d` (stride 1, `same` padding, with bias) as arrays; `wino`: with its F(`mg.WINO_TILE`,3) weights."""
    k = tuple(layer.kernel_size)
    if tuple(layer.stride) != (1, 1) or layer.groups != 1 or layer.bias is None or k[0] != k[1] or k[0] % 2 == 0:
        msg = f"SR CNN MLX: a stride-1, ungrouped, odd square conv with bias, got {layer}"
        raise NotImplementedError(msg)
    if tuple(layer.padding) != (k[0] // 2, k[1] // 2):
        msg = f"SR CNN MLX: `same` padding, got {layer}"
        raise NotImplementedError(msg)
    w, b = layer.weight, layer.bias  # scaled in the conversion (not a torch op on a replay's meta tensors)
    p = {"b": swin.arr(b, dtype, scale=scale)}
    if k == (1, 1):
        p["m"] = swin.arr(w[:, :, 0, 0].t(), dtype, scale=scale)
        return p
    p["w"] = swin.arr(w.permute(0, 2, 3, 1), dtype, scale=scale)
    if wino and k == (3, 3):
        m = mg.WINO_TILE
        cout, cin = int(w.shape[0]), int(w.shape[1])
        p["u"] = swin.derived(((m + 2) ** 2, cin, cout), dtype, lambda: _wino(w, m, scale))
    return p


def prelu(y: mx.array, slope: mx.array) -> mx.array:
    """`PReLU` / `LeakyReLU` with a per-channel (or broadcast) `slope`."""
    return mx.where(y > 0, y, y * slope)


def conv(
    x: mx.array, p: dict[str, mx.array], *, slope: mx.array | None = None, residual: mx.array | None = None
) -> mx.array:
    """`conv(x)`, then `prelu(., slope)` if `slope` is given, then `+ residual`: NHWC, the conv's `params`."""
    if "m" in p:
        y = swin.dense(x, p["m"], p["b"])
    else:
        w = p["w"]
        u = p.get("u")
        if u is not None and x.shape[1] * x.shape[2] > mg.WINO_SMALL_HW:
            from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - Metal only

            if conv_ops.supported(tuple(x.shape), w.shape[0]):
                act = NONE if slope is None else PRELU
                s = None if slope is None else mx.broadcast_to(slope, (w.shape[0],))
                return conv_ops.conv3x3(x, u, p["b"], m=mg.WINO_TILE, act=act, slope=s, residual=residual)
        pad = (w.shape[1] // 2, w.shape[2] // 2)
        y = mg._conv2d(x, w, (1, 1), pad, (1, 1), 1) + p["b"]  # noqa: SLF001 - the translator's dispatch
    if slope is not None:
        y = prelu(y, slope)
    return y if residual is None else y + residual


def set_mode(*, kernels: bool) -> None:
    """The translator's flags `mlx_graph._conv2d` reads, as `GraphNet` sets them for the mode (before each call)."""
    mg._KERNELS, mg._GEMM1X1, mg._HALF_WINO_F32, mg._HALF_WINO_PAD = kernels, True, False, False  # noqa: SLF001
