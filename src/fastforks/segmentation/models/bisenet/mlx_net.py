"""BiSeNet's main head on Apple MLX, hand-written: NHWC, the run dtype, the BatchNorms folded (as `model.network`).

`model._main_head` on one normalised `[1, H, W, 3]` image: the ResNet-18 context path, the two attention refinement
modules (global means, sigmoid gates), the nearest upsamplings as gathers, the feature fusion module, the output conv
and the `align_corners` bilinear upsampling of the 19 logits to the input size, compiled as one graph per input shape.
`kernels`: the translator's direct / Winograd conv kernels where they apply (`shared.mlx_rsu.conv`).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_rsu

from .net import bisenet as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "bisenet-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(mlx_rsu.__file__))


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _mean(x: mx.array) -> mx.array:
    """`F.avg_pool2d(x, x.shape[2:])`: the per-channel mean `[1, 1, 1, C]`, in float32."""
    return x.astype(mx.float32).mean(axis=(1, 2), keepdims=True).astype(x.dtype)


def _maxpool(x: mx.array) -> mx.array:
    """`nn.MaxPool2d(3, stride=2, padding=1)` on NHWC."""
    _, h, w, _ = x.shape
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-float("inf"))
    out = None
    for dy in range(3):
        for dx in range(3):
            t = xp[:, dy : dy + 2 * oh - 1 : 2, dx : dx + 2 * ow - 1 : 2]
            out = t if out is None else mx.maximum(out, t)
    return out


def _nearest(x: mx.array, hw: tuple[int, int]) -> mx.array:
    """`F.interpolate(x, hw, mode="nearest")` on NHWC."""
    h, w = x.shape[1:3]
    if (h, w) == tuple(hw):
        return x
    if (2 * h, 2 * w) == tuple(hw):
        return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
    rows = np.minimum(np.floor(np.arange(hw[0]) * (h / hw[0])).astype(np.int32), h - 1)
    cols = np.minimum(np.floor(np.arange(hw[1]) * (w / hw[1])).astype(np.int32), w - 1)
    return mx.take(mx.take(x, mx.array(rows), axis=1), mx.array(cols), axis=2)


class Network:
    """`_main_head` of a BN-folded `BiSeNet`: `[1, H, W, 3]` float32 -> the logits `[1, H, W, 19]` float32."""

    def __init__(self, net: Any, dtype: str, *, kernels: bool = False) -> None:
        """The arrays of `net` (folded, float32, or the meta skeleton on a cache hit) in `dtype`."""
        self.dtype, self.kernels = getattr(mx, dtype), kernels

        def conv(layer: Any) -> Layer:
            return mlx_rsu.conv(layer, None, self.dtype, kernels=kernels)

        def cbr(m: Any) -> Layer:  # ConvBNReLU (folded)
            c = conv(m.conv)
            return lambda x: _relu(c(x))

        def block(b: Any) -> Layer:
            c1, c2 = conv(b.conv1), conv(b.conv2)
            down = None if b.downsample is None else conv(b.downsample[0])

            def apply(x: mx.array) -> mx.array:
                y = c2(_relu(c1(x)))
                return _relu((x if down is None else down(x)) + y)

            return apply

        def arm(m: Any) -> Layer:
            c, gate = cbr(m.conv), conv(m.conv_atten)

            def apply(x: mx.array) -> mx.array:
                f = c(x)
                return f * mx.sigmoid(gate(_mean(f)))

            return apply

        rn, cp = net.cp.resnet, net.cp
        self.stem = conv(rn.conv1)
        self.layers = [[block(b) for b in getattr(rn, f"layer{i}")] for i in range(1, 5)]
        self.arm16, self.arm32 = arm(cp.arm16), arm(cp.arm32)
        self.head16, self.head32, self.avg = cbr(cp.conv_head16), cbr(cp.conv_head32), cbr(cp.conv_avg)
        ffm = net.ffm
        self.fuse, self.fuse1, self.fuse2 = cbr(ffm.convblk), conv(ffm.conv1), conv(ffm.conv2)
        self.out_conv, self.out = cbr(net.conv_out.conv), conv(net.conv_out.conv_out)
        self.compiled = mx.compile(self.forward)

    def forward(self, x: mx.array) -> mx.array:
        """The logits of `x`."""
        hw = x.shape[1:3]
        y = _maxpool(_relu(self.stem(x.astype(self.dtype))))
        feats = []
        for layer in self.layers:
            for blk in layer:
                y = blk(y)
            feats.append(y)
        f8, f16, f32 = feats[1:]
        s16, s8 = f16.shape[1:3], f8.shape[1:3]
        up32 = self.head32(_nearest(self.arm32(f32) + self.avg(_mean(f32)), s16))
        up16 = self.head16(_nearest(self.arm16(f16) + up32, s8))
        feat = self.fuse(mx.concatenate([f8, up16], axis=-1))
        feat = feat + feat * mx.sigmoid(self.fuse2(_relu(self.fuse1(_mean(feat)))))
        logits = self.out(self.out_conv(feat))
        return mlx_rsu.up_like(logits.astype(mx.float32), hw, align=True)

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` compiled for the input shape, evaluated."""
        mlx_rsu.options(kernels=self.kernels)
        y = self.compiled(x)
        mx.eval(y)
        return y


def network(net: Any, dtype: str, *, kernels: bool = False) -> Network:
    """The MLX main head of a BN-folded `BiSeNet` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype, kernels=kernels)
