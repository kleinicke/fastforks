"""RetinaFace (ResNet-50 / MobileNetV1 x 0.25) on Apple MLX, hand-written: NHWC, the run dtype.

The network of `model.Portable.dense`: images `[B, 3, H, W]` RGB float32 in [0, 1] -> the stride 8 / 16 / 32 head
maps `[B, HEAD, ceil(H / s), ceil(W / s)]` float32. BGR times 255 minus the reference mean, then the backbone, `FPN`
(nearest upsampling cropped to the finer map), `SSH` and the merged 1 x 1 heads. Every conv has its BatchNorm folded
at load (`cspnext.mlx_net.folded`) and dispatches as the graph translator's (`graph="portable"`: 1 x 1 convs as GEMMs,
the MobileNet's depthwise ones as the shared depthwise kernel, the others through `mlx_graph._conv2d`); bias,
(Leaky)ReLU and residual fuse into one compiled pass per piece. The stem and each MobileNet conv pair or ResNet
bottleneck are a step, the FPN another, each level's SSH with its head another.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.cspnext import mlx_net as csp

from . import net as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "retinaface-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(csp.__file__))
GRAPH = "portable"  # the translator's conv dispatch (`csp.conv`)
MEAN_BGR = (104.0, 117.0, 123.0)  # fixed: reference input mean (BGR), as `model.MEAN_BGR`


def _act(leaky: float | None) -> Layer:
    if leaky is None:
        return lambda x: x
    if leaky:
        return lambda x: mx.maximum(x, x * leaky)
    return lambda x: mx.maximum(x, 0)


def conv_bn(seq: Any, dtype: mx.Dtype) -> Layer:
    """`net.conv_bn` (`Sequential(conv, bn[, act])`, also the MobileNet's depthwise-separable 6-entry ones)."""
    mods = list(seq.children())
    ops: list[Layer] = []
    i = 0
    while i < len(mods):
        c = csp.conv(csp.folded(mods[i], mods[i + 1]), dtype, GRAPH)
        i += 2
        act = None
        if i < len(mods) and type(mods[i]).__name__ in ("ReLU", "LeakyReLU"):
            act = float(getattr(mods[i], "negative_slope", 0.0))
            i += 1
        ops.append(csp.chain(c, _act(act)))
    return csp.chain(*ops)


def _bottleneck(blk: Any, dtype: mx.Dtype) -> Layer:
    c1, c2, c3 = (
        csp.conv(csp.folded(getattr(blk, f"conv{i}"), getattr(blk, f"bn{i}")), dtype, GRAPH) for i in (1, 2, 3)
    )
    down = None if blk.downsample is None else csp.conv(csp.folded(blk.downsample[0], blk.downsample[1]), dtype, GRAPH)

    def apply(x: mx.array) -> mx.array:
        y = c3(mx.maximum(c2(mx.maximum(c1(x), 0)), 0))
        return mx.maximum(y + (x if down is None else down(x)), 0)

    return apply


def max_pool(x: mx.array) -> mx.array:
    """`nn.MaxPool2d(3, 2, 1)` on NHWC."""
    n, h, w, c = x.shape
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    y = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-math.inf)
    rows = mx.maximum(mx.maximum(y[:, 0 : 2 * oh - 1 : 2], y[:, 1 : 2 * oh : 2]), y[:, 2 : 2 * oh + 1 : 2])
    out = mx.maximum(
        mx.maximum(rows[:, :, 0 : 2 * ow - 1 : 2], rows[:, :, 1 : 2 * ow : 2]), rows[:, :, 2 : 2 * ow + 1 : 2]
    )
    return out.reshape(n, oh, ow, c)


def _up(x: mx.array, like: mx.array) -> mx.array:
    """`net._up`: nearest x2, cropped to `like`'s size."""
    n, h, w, c = x.shape
    y = mx.broadcast_to(x[:, :, None, :, None], (n, h, 2, w, 2, c)).reshape(n, 2 * h, 2 * w, c)
    return y[:, : like.shape[1], : like.shape[2]]


class Network:
    """`RetinaFace.forward` with the runner's input transform: RGB `[B, 3, H, W]` in [0, 1] -> three head maps."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.dtype = dt
        body = net.body
        self.mean = mx.array(MEAN_BGR, mx.float32)
        self.bgr = mx.array([2, 1, 0])
        # a step per piece (a MobileNet conv pair, a bottleneck): one per stage held a stage's transients in one buffer
        if type(body).__name__ == "MobileNet":
            pieces = [[conv_bn(m, dt) for m in s] for s in (body.stage1, body.stage2, body.stage3)]
            first = pieces[0][0]
            pieces[0][0] = lambda x: first(self._input(x))
        else:
            c1 = csp.conv(csp.folded(body.conv1, body.bn1), dt, GRAPH)
            pieces = [
                [_bottleneck(b, dt) for b in layer] for layer in (body.layer1, body.layer2, body.layer3, body.layer4)
            ]
            pieces[0].insert(0, lambda x: max_pool(mx.maximum(c1(self._input(x)), 0)))
            pieces = [pieces[0] + pieces[1], pieces[2], pieces[3]]  # the stride 8 / 16 / 32 outputs
        self.stages = [[csp.stepped(f) for f in stage] for stage in pieces]
        fpn = net.fpn
        o1, o2, o3 = (conv_bn(getattr(fpn, f"output{i}"), dt) for i in (1, 2, 3))
        m1, m2 = conv_bn(fpn.merge1, dt), conv_bn(fpn.merge2, dt)
        sshs = [self._ssh(getattr(net, f"ssh{i}"), dt) for i in (1, 2, 3)]
        heads = [csp.conv(h, dt, GRAPH) for h in net.head]

        def fpn(c3: mx.array, c4: mx.array, c5: mx.array) -> tuple[mx.array, ...]:
            p5 = o3(c5)
            p4 = m2(o2(c4) + _up(p5, c4))
            return m1(o1(c3) + _up(p4, c3)), p4, p5

        # the FPN a step, each level's SSH with its head another (one neck step: R50 float16 376 MB of a 507 MB call)
        self.fpn = csp.stepped(fpn)
        self.levels = [
            csp.stepped(lambda p, h=h, s=s: h(s(p)).astype(mx.float32).transpose(0, 3, 1, 2))
            for h, s in zip(heads, sshs, strict=True)
        ]

    def _input(self, x: mx.array) -> mx.array:
        """RGB NCHW in [0, 1] -> BGR NHWC times 255 minus the mean, in the run dtype."""
        return (mx.take(x, self.bgr, axis=1).transpose(0, 2, 3, 1) * 255 - self.mean).astype(self.dtype)

    @staticmethod
    def _ssh(m: Any, dtype: mx.Dtype) -> Layer:
        c3, c5a, c5b, c7a, c7b = (
            conv_bn(getattr(m, n), dtype) for n in ("conv3X3", "conv5X5_1", "conv5X5_2", "conv7X7_2", "conv7x7_3")
        )

        def apply(x: mx.array) -> mx.array:
            t = c5a(x)
            return mx.maximum(mx.concatenate([c3(x), c5b(t), c7b(c7a(t))], axis=-1), 0)

        return apply

    def forward(self, x: mx.array) -> tuple[mx.array, ...]:
        """The network on `[B, 3, H, W]` float32 (the steps)."""
        outs = []
        for stage in self.stages:
            for step in stage:
                x = step(x)
            outs.append(x)
        return tuple(f(p) for f, p in zip(self.levels, self.fpn(*outs), strict=True))

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """`forward` with its steps, evaluated."""
        csp.options(GRAPH)
        out = csp.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `RetinaFace` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
