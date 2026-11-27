"""SwinIR on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_net`'s blocks): NHWC maps, the run dtype.

The network of `model.Portable` (its relative position biases gathered at load, the shifted-window masks made once per
size); float16 divides the last residual group's final conv and its skip by `model._LAST_SCALE` as Portable does
(the group's output exceeds the float16 range), the final norm's epsilon by its square.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastforks._core.mlx import mx
from fastforks.shared import swin_sr
from fastforks.shared.swin_sr import mlx_net as swin

LAYOUT = "swinir-v1"
SOURCES = (Path(__file__), Path(swin.__file__))

_LAST_SCALE = 64.0  # as `model._LAST_SCALE` (float16 only)


def network(net: Any, dtype: str) -> swin.Network:  # noqa: ANN401 - a SwinIR (ty: its submodules are `Module`s)
    """The MLX network of a SwinIR (float32 or meta skeleton) in `dtype`."""
    dt = getattr(mx, dtype)
    half = dt == mx.float16
    groups = []
    for i, layer in enumerate(net.layers):
        blocks = []
        for m in layer.residual_group.blocks:
            a = m.attn
            bias = swin_sr.relative_bias(a.relative_position_bias_table, a.relative_position_index)
            att = swin.Attention(a.qkv, a.proj, bias, dt, heads=int(a.num_heads), scale=float(a.scale))
            blocks.append(swin.Block(m, att, dt))
        last = half and i == len(net.layers) - 1
        groups.append(swin.Group(blocks, [layer.conv], dt, scale=1 / _LAST_SCALE if last else 1.0))
    return swin.Network(net, groups, dt, norm_eps_scale=1 / _LAST_SCALE**2 if half else 1.0)
