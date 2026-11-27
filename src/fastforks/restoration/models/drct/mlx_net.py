"""DRCT on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_net`'s blocks): NHWC maps, the run dtype.

The network of `model.Portable`: each residual dense group runs five Swin layers on the growing concatenation of its
input and the previous layers' outputs, each followed by its 1 x 1 `adjust` conv (a product on the channels, upstream:
around a token unembedding) and a LeakyReLU (not the last), then `0.2 * x5 + x`. The relative position biases are
gathered at load, the shifted-window masks made once per size. float16 needs no rescaling (as Portable).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared import swin_sr
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from .net import DRCT

LAYOUT = "drct-v1"
SOURCES = (Path(__file__), Path(swin.__file__))

_LAYERS = 5


@swin.stepped
def _grow(y: mx.array, feats: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """`cat(feats, lrelu(adjust(y)))`: the next layer's input."""
    return mx.concatenate([feats, swin.lrelu(swin.dense(y, w, b), 0.2)], axis=-1)


@swin.stepped
def _close(y: mx.array, x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """`0.2 * adjust5(y) + x`: the group's output."""
    return swin.dense(y, w, b) * 0.2 + x


class DenseGroup:
    """An `RDG`: `[B, H, W, C]` -> the same."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - the vendored RDG
        """The arrays of the RDG `m` in `dtype`."""
        self.blocks = []
        self.adjust = []
        for k in range(1, _LAYERS + 1):
            b = m.get_submodule(f"swin{k}")
            a = b.attn
            bias = swin_sr.relative_bias(a.relative_position_bias_table, a.relative_position_index)
            att = swin.Attention(a.qkv, a.proj, bias, dtype, heads=int(a.num_heads), scale=float(a.scale))
            self.blocks.append(swin.Block(b, att, dtype))
            c = m.get_submodule(f"adjust{k}")  # a 1 x 1 Conv2d
            self.adjust.append((swin.arr(c.weight[:, :, 0, 0].t(), dtype), swin.arr(c.bias, dtype)))

    def __call__(self, x: mx.array) -> mx.array:
        """The group's output."""
        feats = x
        for k in range(_LAYERS - 1):
            feats = _grow(self.blocks[k](feats), feats, *self.adjust[k])
        return _close(self.blocks[-1](feats), x, *self.adjust[-1])


def network(net: DRCT, dtype: str) -> swin.Network:
    """The MLX network of a DRCT (float32 or meta skeleton) in `dtype`."""
    dt = getattr(mx, dtype)
    return swin.Network(net, [DenseGroup(layer, dt) for layer in net.layers], dt)
