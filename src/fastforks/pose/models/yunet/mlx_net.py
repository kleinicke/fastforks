"""YuNet on Apple MLX, hand-written: NHWC, the run dtype.

The network of `model.Portable.dense` after its padding: BGR `[B, 3, H, W]` float32 in 0..255 (H, W multiples of 32)
-> the stride 8 / 16 / 32 head maps `[B, 16, H / s, W / s]` float32. Each `ConvDPUnit` is its 1 x 1 conv as a GEMM
and its 3 x 3 depthwise conv (the shared depthwise kernel) with the BatchNorm folded in at load
(`cspnext.mlx_net.folded`), the convs dispatched as the graph translator's (`graph="portable"`); 2 x 2 max pooling as
the maximum over a reshaped window, the neck's nearest x2 as a broadcast. The stem, the second stage and each of the
stride 8 / 16 / 32 stages are a step, the neck with the heads another.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.cspnext import mlx_net as csp

from . import net as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "yunet-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(csp.__file__))
GRAPH = "portable"  # the translator's conv dispatch (`csp.conv`)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def unit(u: Any, dtype: mx.Dtype) -> Layer:
    """`ConvDPUnit`: 1 x 1 conv, depthwise 3 x 3 conv (BatchNorm folded) and ReLU, or both convs alone (heads)."""
    c1 = csp.conv(u.conv1, dtype, GRAPH)
    if u.bn is None:
        c2 = csp.conv(u.conv2, dtype, GRAPH)
        return lambda x: c2(c1(x))
    c2 = csp.conv(csp.folded(u.conv2, u.bn), dtype, GRAPH)
    return lambda x: _relu(c2(c1(x)))


def pool2(x: mx.array) -> mx.array:
    """`F.max_pool2d(x, 2)` on NHWC (even sizes)."""
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def up2(x: mx.array) -> mx.array:
    """Nearest x2 on NHWC."""
    n, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None], (n, h, 2, w, 2, c)).reshape(n, 2 * h, 2 * w, c)


class Network:
    """`YuNet.forward`: BGR `[B, 3, H, W]` float32 in 0..255 -> three head maps `[B, 16, H / s, W / s]` float32."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        bb = net.backbone
        stem_conv = csp.conv(csp.folded(bb.model0.conv1, bb.model0.bn1), dt, GRAPH)
        stem_unit = unit(bb.model0.conv2, dt)
        blocks = [
            csp.chain(unit(b.conv1, dt), unit(b.conv2, dt))
            for b in (bb.model1, bb.model2, bb.model3, bb.model4, bb.model5)
        ]

        # a step per stage (one step for the backbone held the full-resolution stem's transients with the rest)
        self.stem = csp.stepped(lambda x: pool2(stem_unit(_relu(stem_conv(x.transpose(0, 2, 3, 1).astype(dt))))))
        self.stage2 = csp.stepped(lambda y: pool2(blocks[1](blocks[0](y))))
        self.c3, self.c4 = csp.stepped(blocks[2]), csp.stepped(lambda y: blocks[3](pool2(y)))
        self.c5 = csp.stepped(lambda y: blocks[4](pool2(y)))

        lat = [unit(u, dt) for u in net.neck.lateral_convs]
        head = net.bbox_head
        shared = (
            [csp.chain(*(unit(u, dt) for u in seq)) for seq in head.multi_level_share_convs]
            if head.multi_level_share_convs is not None
            else [lambda x: x] * tnet.LEVELS
        )
        outs = [unit(u, dt) for u in head.heads]

        def neck(c3: mx.array, c4: mx.array, c5: mx.array) -> tuple[mx.array, ...]:
            p5 = lat[2](c5)
            p4 = lat[1](c4 + up2(p5))
            p3 = lat[0](c3 + up2(p4))
            return tuple(
                o(s(p)).astype(mx.float32).transpose(0, 3, 1, 2)
                for o, s, p in zip(outs, shared, (p3, p4, p5), strict=True)
            )

        self.neck = csp.stepped(neck)

    def forward(self, x: mx.array) -> tuple[mx.array, ...]:
        """The network on `[B, 3, H, W]` float32 (the steps)."""
        c3 = self.c3(self.stage2(self.stem(x)))
        c4 = self.c4(c3)
        return self.neck(c3, c4, self.c5(c4))

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """`forward` with its steps, evaluated."""
        csp.options(GRAPH)
        out = csp.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: Any, dtype: str) -> Network:
    """The MLX network of a `YuNet` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
