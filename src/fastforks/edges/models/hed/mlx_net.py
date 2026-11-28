"""HED (ControlNet 1.1) on Apple MLX, hand-written (`fastforks.shared.mlx_conv`): NHWC, the run dtype.

The network of `net.hed`: the input minus its learned mean, five VGG-style blocks (3, 3, 3 / 2, 2 convs with ReLU, a
2 x 2 max pool before blocks 2-5), a 1-channel projection of each block's output. One compiled step per block; the
sides leave as float32 `[B, h, w, 1]` maps (the runner resizes and averages them). In float16 the 3 x 3 convolutions
MLX would run as Winograd run in float32, as the graph translation did (`half_wino_f32`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared import mlx_conv
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net import ControlNetHED_Apache2

LAYOUT = "hed-v1"
SOURCES = (Path(__file__), Path(mlx_conv.__file__), Path(__file__).with_name("net") / "hed.py")


class Network:
    """`ControlNetHED_Apache2.forward`: `[B, H, W, 3]` levels 0..255 -> its five side maps `[B, h, w, 1]` float32."""

    def __init__(self, net: ControlNetHED_Apache2, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        n: Any = net
        mean = swin.arr(n.norm.reshape(3), dt)
        blocks = [n.block1, n.block2, n.block3, n.block4, n.block5]
        self.half = dtype == "float16"
        self.steps = [self._step(i, [mlx_conv.Conv(c, dt) for c in b.convs], mlx_conv.Conv(b.projection, dt), mean)
                      for i, b in enumerate(blocks)]  # fmt: skip

    @staticmethod
    def _step(i: int, convs: list[mlx_conv.Conv], proj: mlx_conv.Conv, mean: mx.array) -> Callable[..., Any]:
        """Block `i` and its projection as one compiled step: map in -> (map out, side map)."""

        def step(x: mx.array) -> tuple[mx.array, mx.array]:
            if i:
                n, h, w, c = x.shape
                y = x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))  # MaxPool2d(2, 2)
            else:
                y = x.astype(mean.dtype) - mean
            for conv in convs:
                y = mx.maximum(conv(y), 0)
            return y, proj(y).astype(mx.float32)

        return mx.compile(step)

    def __call__(self, x: mx.array) -> list[mx.array]:
        """The side maps of `x`, evaluated."""
        sides = []
        with mlx_conv.mode(half_wino_f32=self.half):
            for step in self.steps:
                x, side = step(x)
                mx.eval(x, side)
                sides.append(side)
        return sides


def network(net: ControlNetHED_Apache2, dtype: str) -> Network:
    """The MLX network of a HED (float32, or the meta skeleton) in `dtype`."""
    return Network(net, dtype)
