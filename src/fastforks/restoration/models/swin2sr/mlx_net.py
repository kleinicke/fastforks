"""Swin2SR on Apple MLX, hand-written (`fastforks.shared.swin_sr.mlx_net`'s blocks): NHWC maps, the run dtype.

The network of `model.Portable`: Swin V2's post-norm blocks and cosine attention (q and k normalised in float32), the
continuous position bias (`16 * sigmoid(cpb_mlp(table))`) and the clamped logit scale computed once at load, q / v
biases as one `qkv` bias. The last residual group's output is scaled by `model._LAST_SCALE` in every dtype as Portable
does (it overflows float16), the final norm's epsilon by its square.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from fastforks._core.mlx import mx
from fastforks.shared.swin_sr import mlx_net as swin

LAYOUT = "swin2sr-v1"
SOURCES = (Path(__file__), Path(swin.__file__))

_LAST_SCALE = 1 / 32  # as `model._LAST_SCALE`


def _attention(a: Any, dt: mx.Dtype) -> swin.Attention:  # noqa: ANN401 - the vendored WindowAttention
    """A Swin V2 `WindowAttention`'s arrays (bias and logit scale from its tensors: meta on a cache hit)."""
    n = a.relative_position_index.shape[0]
    if a.logit_scale.is_meta:  # a replay reads shapes only (arithmetic on meta tensors imports `torch._dynamo`)
        bias = torch.empty((int(a.num_heads), n, n), device="meta")
        qkv_bias = None if a.q_bias is None else torch.empty(3 * a.q_bias.shape[0], device="meta")
        heads = int(a.num_heads)
        return swin.Attention(a.qkv, a.proj, bias, dt, heads=heads, scale=1.0, qkv_bias=qkv_bias, cosine=a.logit_scale)
    with torch.no_grad():
        table = a.cpb_mlp(a.relative_coords_table).view(-1, a.num_heads)
        bias = 16 * torch.sigmoid(table[a.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1))
        cosine = torch.clamp(a.logit_scale, max=math.log(1.0 / 0.01)).exp()
        qkv_bias = None
        if a.q_bias is not None:
            qkv_bias = torch.cat((a.q_bias, torch.zeros_like(a.v_bias), a.v_bias))
    return swin.Attention(a.qkv, a.proj, bias, dt, heads=int(a.num_heads), scale=1.0, qkv_bias=qkv_bias, cosine=cosine)


def network(net: Any, dtype: str) -> swin.Network:  # noqa: ANN401 - a Swin2SR (ty: its submodules are `Module`s)
    """The MLX network of a Swin2SR (float32 or meta skeleton) in `dtype`."""
    dt = getattr(mx, dtype)
    groups = []
    for i, layer in enumerate(net.layers):
        blocks = [swin.Block(m, _attention(m.attn, dt), dt, post=True) for m in layer.residual_group.blocks]
        convs = [layer.conv] if layer.patch_embed.proj is None else [layer.conv, layer.patch_embed.proj]
        last = i == len(net.layers) - 1
        groups.append(swin.Group(blocks, convs, dt, scale=_LAST_SCALE if last else 1.0))
    return swin.Network(net, groups, dt, norm_eps_scale=_LAST_SCALE**2)
