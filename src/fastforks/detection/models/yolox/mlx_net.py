"""YOLOX on Apple MLX, written in `mlx.core` operations (`MlxYOLOX(net, dtype, graph)`): canvases -> head maps.

The maths of `net.YOLOX.forward`, channels last, on the network with its batch norms folded, from the layers of
`fastforks.shared.cspnext.mlx_net` (SiLU in the run dtype, the SPP's max pools): the Focus stem, CSPDarknet, the PAFPN
and the decoupled head in the run dtype, the maps returned float32. The convs dispatch as the graph translator's
(`graph`: "portable" or "kernels", see `cspnext.mlx_net`). Written out instead of replayed from an exported graph:
the export and its translation were most of the first call.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks._core.patch import base_type
from fastforks.shared.cspnext import mlx_net as csp

if TYPE_CHECKING:
    from fastforks.shared.cspnext.mlx_net import Layer

__all__ = ["LAYOUT", "SOURCES", "MlxYOLOX"]

LAYOUT = "yolox-v1"
"""The conversion cache layout of `MlxYOLOX`'s arrays (`fastforks.shared.mlx_port.recorded`; both graphs alike)."""

SOURCES = (Path(__file__), Path(csp.__file__), Path(__file__).with_name("net.py"))
"""The code the cached arrays come from: this builder, the shared layers, the checkpoint conversion."""


def _csp_layer(layer: Any, d: mx.Dtype, graph: str) -> Layer:
    """`CSPLayer`: conv3(cat(bottlenecks(conv1(x)), conv2(x)))."""
    conv1, conv2, conv3 = (csp.conv_act(c, d, graph) for c in (layer.conv1, layer.conv2, layer.conv3))
    blocks = [(csp.conv_act(b.conv1, d, graph), csp.conv_act(b.conv2, d, graph), b.add) for b in layer.m]

    def apply(x: mx.array) -> mx.array:
        y = conv1(x)
        for first, second, add in blocks:
            z = second(first(y))
            y = z + y if add else z
        return conv3(mx.concatenate((y, conv2(x)), axis=-1))

    return apply


def _stage(stage: Any, d: mx.Dtype, graph: str) -> Layer:
    """A `dark` stage: the stride-2 conv, the SPP (last stage), the `CSPLayer`."""
    build = {"CSPLayer": _csp_layer, "SPP": csp.spp}
    return csp.chain(*(build.get(base_type(m).__name__, csp.conv_act)(m, d, graph) for m in stage))


class MlxYOLOX:
    """NHWC canvases in 0..255 (float32) -> float32 (box + objectness, class logits) NHWC maps per level, flat."""

    def __init__(self, net: Any, dtype: str, graph: str) -> None:
        """The MLX layers of `net` (a `YOLOX` with its batch norms folded) in `dtype`, convs as `graph` dispatches."""
        d = mx.float16 if dtype == "float16" else mx.float32
        pa, bb, h = net.backbone, net.backbone.backbone, net.head
        stem = csp.conv_act(bb.stem.conv, d, graph)
        dark = [_stage(getattr(bb, f"dark{i}"), d, graph) for i in range(2, 6)]
        lateral, reduce = csp.conv_act(pa.lateral_conv0, d, graph), csp.conv_act(pa.reduce_conv1, d, graph)
        bu2, bu1 = csp.conv_act(pa.bu_conv2, d, graph), csp.conv_act(pa.bu_conv1, d, graph)
        p4, p3, n3, n4 = (_csp_layer(m, d, graph) for m in (pa.C3_p4, pa.C3_p3, pa.C3_n3, pa.C3_n4))
        levels = [
            (
                csp.conv_act(h.stems[i], d, graph),
                csp.chain(*(csp.conv_act(m, d, graph) for m in h.reg_convs[i])),
                csp.conv(h.reg_obj_preds[i], d, graph),
                csp.chain(*(csp.conv_act(m, d, graph) for m in h.cls_convs[i])),
                csp.conv(h.cls_preds[i], d, graph),
            )
            for i in range(len(h.stems))
        ]

        def focus(x: mx.array) -> mx.array:
            x = x.astype(d)
            # Focus: the four 2 x 2 phases (top-left, bottom-left, top-right, bottom-right), channel-concatenated
            x = stem(mx.concatenate((x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]), axis=-1))
            return dark[1](dark[0](x))

        def _top4(c4: mx.array, c5: mx.array) -> tuple[mx.array, mx.array]:
            f5 = lateral(c5)
            return f5, reduce(p4(mx.concatenate((csp.up2(f5), c4), axis=-1)))

        def head(f: mx.array, level: tuple[Layer, ...]) -> tuple[mx.array, mx.array]:
            head_stem, reg_tower, reg_obj, cls_tower, cls = level
            y = head_stem(f)
            return reg_obj(reg_tower(y)).astype(mx.float32), cls(cls_tower(y)).astype(mx.float32)

        # each stage, neck block and head level its own compiled step (`csp.stepped`: bounded transients)
        first, s4, s5 = csp.stepped(focus), csp.stepped(dark[2]), csp.stepped(dark[3])
        top4 = csp.stepped(_top4)
        top3 = csp.stepped(lambda c3, f4: p3(mx.concatenate((csp.up2(f4), c3), axis=-1)))
        bottom4 = csp.stepped(lambda o3, f4: n3(mx.concatenate((bu2(o3), f4), axis=-1)))
        bottom5 = csp.stepped(lambda o4, f5: n4(mx.concatenate((bu1(o4), f5), axis=-1)))
        heads = [csp.stepped(lambda f, level=level: head(f, level)) for level in levels]

        def forward(x: mx.array) -> tuple[mx.array, ...]:
            c3 = first(x)
            c4 = s4(c3)
            f5, f4 = top4(c4, s5(c4))
            o3 = top3(c3, f4)
            o4 = bottom4(o3, f4)
            o5 = bottom5(o4, f5)
            return tuple(m for f, h in zip((o3, o4, o5), heads, strict=True) for m in h(f))

        self.forward = forward

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """The six head maps of canvases `[B, H, W, 3]` in 0..255."""
        return csp.steps(self.forward, x)
