"""M-LSD large (MobileNetV2 U-Net) on Apple MLX, hand-written (`fastforks.shared.mlx_conv`): NHWC, the run dtype.

The network of `net.mbv2_mlsd_large`: a MobileNetV2 backbone to stride 16 (five taps), four fusion stages (a 1x1 on
each of the two inputs, the deeper one upsampled 2x bilinear, `align_corners=True`, the two concatenated, then two 3x3
convolutions with a residual) and a head (a dilated 3x3, a 3x3, a 1x1 of which only the last 9 of 16 channels are
kept). Every BatchNorm is folded into the convolution before it in the conversion. One compiled forward.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared import mlx_conv

if TYPE_CHECKING:
    from .net import MobileV2_MLSD_Large

LAYOUT = "mlsd-v2"
SOURCES = (Path(__file__), Path(mlx_conv.__file__), Path(__file__).with_name("net") / "mbv2_mlsd_large.py")
STRIDE2 = 2
TAPS = (1, 3, 6, 10, 13)  # the backbone features whose outputs the decoder reads


def _relu6(x: mx.array) -> mx.array:
    return mx.clip(x, 0, 6)


class Network:
    """`MobileV2_MLSD_Large.forward`: `[B, H, W, 4]` -> the center and displacement maps `[B, H/2, W/2, 9]`."""

    def __init__(self, net: MobileV2_MLSD_Large, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`, the norms folded in."""
        dt = getattr(mx, dtype)
        n: Any = net
        conv = functools.partial(mlx_conv.Conv, dtype=dt)

        def cbr(m: Any) -> tuple[mlx_conv.Conv, bool]:  # noqa: ANN401 - torch module
            """A ConvBNReLU: the folded convolution, whether it pads (stride 2)."""
            return conv(m[0], bn=m[1]), m.stride == STRIDE2

        feats = n.backbone.features
        self.stem = cbr(feats[0])
        self.blocks: list[tuple[list[tuple[mlx_conv.Conv, bool]], mlx_conv.Conv, bool]] = []
        for f in feats[1 : TAPS[-1] + 1]:
            layers = list(f.conv)
            body = [cbr(m) for m in layers[:-2]]
            self.blocks.append((body, conv(layers[-2], bn=layers[-1]), bool(f.use_res_connect)))
        self.stages = []
        for a, b in ((n.block15, n.block16), (n.block17, n.block18), (n.block19, n.block20), (n.block21, n.block22)):
            self.stages.append(
                (
                    conv(a.conv1[0], bn=a.conv1[1]),
                    conv(a.conv2[0], bn=a.conv2[1]),
                    a.upscale,
                    conv(b.conv1[0], bn=b.conv1[1]),
                    conv(b.conv2[0], bn=b.conv2[1]),
                )
            )
        c = n.block23
        self.head = (
            conv(c.conv1[0], bn=c.conv1[1]),
            conv(c.conv2[0], bn=c.conv2[1]),
            conv(c.conv3),  # all 16 outputs: 9 is a ragged width for MLX's GEMM, the 9 kept are sliced after
        )
        self._coarse = mx.compile(self._run_coarse)
        self._front = mx.compile(self._run_front)
        self._back = mx.compile(self._run_back)

    @staticmethod
    def _cbr(layer: tuple[mlx_conv.Conv, bool], x: mx.array) -> mx.array:
        conv, pad = layer
        if pad:  # TFLite's padding: one row and column at the bottom and right
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return _relu6(conv(x))

    def _fuse(self, stage: int, deep: mx.array, skip: mx.array) -> mx.array:
        """The two inputs of fusion stage `stage` upsampled and concatenated (the `a` map first)."""
        up_b, up_a, upscale, _, _ = self.stages[stage]
        b = mx.maximum(up_b(deep), 0)
        a = mx.maximum(up_a(skip), 0)
        if upscale:
            b = mlx_conv.resize(b, (b.shape[1] * 2, b.shape[2] * 2), align_corners=True)
        return mx.concatenate([a, b], axis=-1)

    def _convs(self, stage: int, x: mx.array) -> mx.array:
        """The two residual 3x3 convolutions of fusion stage `stage` on its concatenated input."""
        _, _, _, b1, b2 = self.stages[stage]
        x = mx.maximum(b1(x), 0) + x
        return mx.maximum(b2(x), 0)

    def _run_coarse(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The backbone and the first three fusion stages: the deep map (stride 2) and the finest tap."""
        x = self._cbr(self.stem, x)
        taps = []
        for i, (body, project, res) in enumerate(self.blocks, 1):
            y = x
            for layer in body:
                y = self._cbr(layer, y)
            y = project(y)
            x = x + y if res else y
            if i in TAPS:
                taps.append(x)
        c1, c2, c3, c4, c5 = taps
        x = c5
        for stage, skip in enumerate((c4, c3, c2)):
            x = self._convs(stage, self._fuse(stage, x, skip))
        return x, c1

    def _run_front(self, deep: mx.array, c1: mx.array) -> mx.array:
        return self._fuse(3, deep, c1)

    def _run_back(self, x: mx.array) -> mx.array:
        h1, h2, h3 = self.head
        x = self._convs(3, x)
        return h3(mx.maximum(h2(mx.maximum(h1(x), 0)), 0))[..., 7:]

    def __call__(self, x: mx.array) -> mx.array:
        """The maps of `x`, evaluated.

        Three steps, each evaluated before the next: MLX keeps a step's temporaries until its command buffers have
        run, and the finest stage's Winograd 3x3 (128 -> 128 channels at the half resolution) alone needs about
        6x its input; one compiled forward peaked at 1.55x the graph port's transient (float16) for the same speed.
        """
        with mlx_conv.mode():
            deep, c1 = self._coarse(x.astype(self.stem[0].w.dtype))
            mx.eval(deep, c1)
            cat = self._front(deep, c1)
            mx.eval(cat)
            y = self._back(cat)
            mx.eval(y)
        return y


def network(net: MobileV2_MLSD_Large, dtype: str) -> Network:
    """The MLX network of an M-LSD large (float32, or the meta skeleton) in `dtype`."""
    return Network(net, dtype)
