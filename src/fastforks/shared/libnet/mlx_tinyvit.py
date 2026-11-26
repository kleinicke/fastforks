"""TinyViT (timm `TinyVit`, 5M-21M) written in MLX, channels last.

The exported-graph translation took about 1.1 s of the first call (export, translation, segment tuning); written out,
the first call is `mx.compile`'s trace and the shader builds only. Every `ConvNorm` folds its batch norm into the
convolution; the window attention is MLX's fused attention with the learned relative-position bias as its additive
mask (gathered once, at load).
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from fastforks._core.mlx import attention, mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _linear, _norm
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv
from fastforks.shared.libnet.mlx_cnn import _gelu_f32 as _gelu

if TYPE_CHECKING:
    from collections.abc import Callable


def _cn(layer: Any, dtype: mx.Dtype) -> Callable:
    """A timm `ConvNorm`: conv (no bias) + folded batch norm."""
    return _conv(layer.conv, layer.bn, dtype)


def _mbconv(block: Any, dtype: mx.Dtype) -> Callable:
    path = _chain(_cn(block.conv1, dtype), _gelu, _cn(block.conv2, dtype), _gelu, _cn(block.conv3, dtype))
    return lambda x: _gelu(x + path(x))


def _merge(layer: Any, dtype: mx.Dtype) -> Callable:
    """`PatchMerging`: 1x1, depthwise 3x3 (stride 2), 1x1, GELU between."""
    return _chain(_cn(layer.conv1, dtype), _gelu, _cn(layer.conv2, dtype), _gelu, _cn(layer.conv3, dtype))


def _attention(attn: Any, window: int, dtype: mx.Dtype) -> Callable:
    """Window attention of `[windows, N, C]` tokens: LayerNorm, qkv, fused attention with the position bias, proj."""
    norm, qkv, proj = _norm(attn.norm, dtype), _linear(attn.qkv, dtype), _linear(attn.proj, dtype)
    heads, kd, vd = attn.num_heads, attn.key_dim, attn.val_dim
    bias = to_mx(attn.attention_biases.detach()[:, attn.attention_bias_idxs]).astype(dtype)  # [heads, N, N]
    mx.eval(bias)
    scale = kd**-0.5
    if bias.shape[-1] != window * window:
        msg = "TinyViT attention resolution differs from its window size"
        raise NotImplementedError(msg)

    def apply(x: mx.array) -> mx.array:
        b, n, _ = x.shape
        t = qkv(norm(x)).reshape(b, n, heads, 2 * kd + vd).transpose(0, 2, 1, 3)
        q, k, v = t[..., :kd], t[..., kd : 2 * kd], t[..., 2 * kd :]
        o = attention(q, k, v, scale=scale, mask=bias)
        return proj(o.transpose(0, 2, 1, 3).reshape(b, n, heads * vd))

    return apply


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    ws = block.window_size
    attn = _attention(block.attn, ws, dtype)
    local = _cn(block.local_conv, dtype)
    m = block.mlp
    mlp = _chain(_norm(m.norm, dtype), _linear(m.fc1, dtype), _gelu, _linear(m.fc2, dtype))

    def windowed(x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        pb, pr = -h % ws, -w % ws
        y = mx.pad(x, [(0, 0), (0, pb), (0, pr), (0, 0)]) if pb or pr else x
        nh, nw = (h + pb) // ws, (w + pr) // ws
        y = y.reshape(b, nh, ws, nw, ws, c).transpose(0, 1, 3, 2, 4, 5).reshape(b * nh * nw, ws * ws, c)
        y = attn(y)
        y = y.reshape(b, nh, nw, ws, ws, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, nh * ws, nw * ws, c)
        return y[:, :h, :w] if pb or pr else y

    def apply(x: mx.array) -> mx.array:
        x = local(x + windowed(x))
        return x + mlp(x)

    return apply


def tinyvit(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `TinyVit`: NHWC images -> last stage map, pooled `[B, C]` -> logits."""
    head = net.head
    if not isinstance(head.pre_logits, nn.Identity):
        msg = "TinyViT with a hidden head layer has no MLX port"
        raise NotImplementedError(msg)
    pe = net.patch_embed
    ops: list[Callable] = [_cn(pe.conv1, dtype), _gelu, _cn(pe.conv2, dtype)]
    for stage in net.stages:
        if type(stage).__name__ == "ConvLayer":
            ops.extend(_mbconv(block, dtype) for block in stage.blocks)
            continue
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_merge(stage.downsample, dtype))
        ops.extend(_block(block, dtype) for block in stage.blocks)
    return _chain(*ops), _chain(_norm(head.norm, dtype), _linear(head.fc, dtype))
