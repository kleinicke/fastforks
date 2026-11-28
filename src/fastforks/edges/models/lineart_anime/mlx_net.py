"""The anime line art U-Net (pix2pix generator) on Apple MLX, hand-written (`fastforks.shared.mlx_conv`): NHWC.

Eight stride-2 4 x 4 convolutions down, eight transposed ones up (`mlx_conv.ConvT`), the encoder maps concatenated
onto the decoder maps, instance norms (float32 statistics) between, tanh at the end. The biases of the layers a norm
follows are dropped in the conversion (the norm subtracts them again).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared import mlx_conv

if TYPE_CHECKING:
    from .net import UnetGenerator

LAYOUT = "lineart-anime-v1"
SOURCES = (Path(__file__), Path(mlx_conv.__file__), Path(__file__).with_name("net") / "unet.py")


class Network:
    """`UnetGenerator.forward`: `[B, H, W, 3]` in [-1, 1] -> `[B, H, W, 1]` in [-1, 1] (H, W multiples of 256)."""

    def __init__(self, net: UnetGenerator, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        # the nested skip blocks, outermost first: (down conv, up conv)
        self.levels: list[tuple[mlx_conv.Conv, mlx_conv.ConvT]] = []
        block: Any = net.model
        while block is not None:
            layers = list(block.model)
            down = next(m for m in layers if type(m).__name__ == "Conv2d")
            up = next(m for m in layers if type(m).__name__ == "ConvTranspose2d")
            outermost = block.outermost
            self.levels.append(
                (
                    mlx_conv.Conv(down, dt, bias=outermost),
                    mlx_conv.ConvT(up, dt, bias=outermost),
                )
            )
            block = next((m for m in layers if hasattr(m, "outermost")), None)

    def _level(self, i: int, x: mx.array) -> mx.array:
        """Level `i` (0: the outermost, the last: the innermost) of the U-Net."""
        down, up = self.levels[i]
        last = i == len(self.levels) - 1
        if i == 0:
            h = self._level(1, down(x))
            return mx.tanh(up(mx.maximum(h, 0)))
        h = down(mx.maximum(x, 0.2 * x))
        if not last:
            h = self._level(i + 1, mlx_conv.instance_norm(h))
        h = mlx_conv.instance_norm(up(mx.maximum(h, 0)))
        mx.eval(h)  # one level at a time: MLX holds temporaries until a step ends
        return mx.concatenate([x, h], axis=-1)

    def _run(self, x: mx.array) -> mx.array:
        with mlx_conv.mode():
            return self._level(0, x)

    def __call__(self, x: mx.array) -> mx.array:
        """The line map of the NHWC image `x`."""
        y = self._run(x)
        mx.eval(y)
        return y


def network(net: UnetGenerator, dtype: str) -> Network:
    """The MLX network of a U-Net (float32, or the meta skeleton) in `dtype`."""
    return Network(net, dtype)
