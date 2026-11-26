"""EfficientFormer (timm `EfficientFormer`, L1-L7) written in MLX, channels last.

The exported-graph translation spent about a second of the first call on export, translation and segment tuning;
written out, the first call is `mx.compile`'s trace and the shader builds only. Every batch norm folds into the
convolution it follows and every layer scale into the layer before it; the pooling mixer `x + g * (pool(x) - x)` is
one blend. The last stage's attention is MLX's fused attention with the learned position bias as its additive mask
(gathered once, at load); the distilled heads average into one linear layer.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks._core.mlx import attention, mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _linear, _norm
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv
from fastforks.shared.libnet.mlx_cnn import _gelu_f32 as _gelu

if TYPE_CHECKING:
    from collections.abc import Callable


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _avg3(x: mx.array) -> mx.array:
    """3x3 average pool, stride 1, padding 1 not counted (`count_include_pad=False`), as separable sums."""
    h, w = x.shape[1], x.shape[2]
    y = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    y = y[:, :h] + y[:, 1 : h + 1] + y[:, 2 : h + 2]
    y = y[:, :, :w] + y[:, :, 1 : w + 1] + y[:, :, 2 : w + 2]
    rows = 3 - (mx.arange(h) == 0) - (mx.arange(h) == h - 1)  # 2 at the borders, 3 inside (1 for a single row)
    cols = 3 - (mx.arange(w) == 0) - (mx.arange(w) == w - 1)
    inv = (1 / (rows[:, None] * cols[None, :]).astype(mx.float32)).astype(x.dtype)
    return y * inv[None, :, :, None]


def _scale(layer: Any) -> torch.Tensor:
    return layer.gamma.detach().flatten()


def _conv_mlp(m: Any, gamma: torch.Tensor, dtype: mx.Dtype) -> Callable:
    """`ConvMlpWithNorm` (1x1 + BN, GELU, 1x1 + BN) with the layer scale folded into the second 1x1."""
    with torch.no_grad():
        fc1 = torch.nn.utils.fuse_conv_bn_eval(m.fc1, m.norm1)
        fc2 = torch.nn.utils.fuse_conv_bn_eval(m.fc2, m.norm2)
    if not isinstance(m.act, nn.GELU):
        msg = "EfficientFormer MLP without GELU has no MLX port"
        raise NotImplementedError(msg)
    return _chain(_linear(fc1, dtype), _gelu, _linear(fc2, dtype, gamma))


def _block2d(block: Any, dtype: mx.Dtype) -> Callable:
    """`MetaBlock2d`: `x + g1 * (pool(x) - x)`, then `x + g2 * mlp(x)`."""
    g1 = to_mx(_scale(block.ls1)).astype(dtype)
    mx.eval(g1)
    mlp = _conv_mlp(block.mlp, _scale(block.ls2), dtype)

    def apply(x: mx.array) -> mx.array:
        x = x + g1 * (_avg3(x) - x)
        return x + mlp(x)

    return apply


def _attention(attn: Any, gamma: torch.Tensor, dtype: mx.Dtype) -> Callable:
    """Token attention `[B, N, C]` with the position bias as mask; the layer scale folds into `proj`."""
    qkv, proj = _linear(attn.qkv, dtype), _linear(attn.proj, dtype, gamma)
    heads, kd, vd = attn.num_heads, attn.key_dim, attn.val_dim
    bias = to_mx(attn.attention_biases.detach()[:, attn.attention_bias_idxs]).astype(dtype)  # [heads, N, N]
    mx.eval(bias)
    n_bias, scale = bias.shape[-1], attn.scale

    def apply(x: mx.array) -> mx.array:
        b, n, _ = x.shape
        if n != n_bias:
            msg = f"EfficientFormer attention needs {n_bias} tokens (got {n})"
            raise ValueError(msg)
        t = qkv(x).reshape(b, n, heads, 2 * kd + vd).transpose(0, 2, 1, 3)
        q, k, v = t[..., :kd], t[..., kd : 2 * kd], t[..., 2 * kd :]
        o = attention(q, k, v, scale=scale, mask=bias)
        return proj(o.transpose(0, 2, 1, 3).reshape(b, n, heads * vd))

    return apply


def _block1d(block: Any, dtype: mx.Dtype) -> Callable:
    """`MetaBlock1d`: pre-norm attention and MLP on tokens, the layer scales folded."""
    m = block.mlp
    if not isinstance(m.norm, nn.Identity) or not isinstance(m.act, nn.GELU):
        msg = "EfficientFormer token MLP with a norm or without GELU has no MLX port"
        raise NotImplementedError(msg)
    attn = _chain(_norm(block.norm1, dtype), _attention(block.token_mixer, _scale(block.ls1), dtype))
    mlp = _chain(_norm(block.norm2, dtype), _linear(m.fc1, dtype), _gelu, _linear(m.fc2, dtype, _scale(block.ls2)))

    def apply(x: mx.array) -> mx.array:
        x = x + attn(x)
        return x + mlp(x)

    return apply


def _flat(x: mx.array) -> mx.array:
    """`Flat`: `[B, H, W, C]` -> `[B, H * W, C]` tokens (row-major, as torch's `flatten(2).transpose(1, 2)`)."""
    return x.reshape(x.shape[0], -1, x.shape[-1])


def _layer(m: Any, dtype: mx.Dtype) -> Callable:
    kind = type(m).__name__
    if kind == "MetaBlock2d":
        return _block2d(m, dtype)
    if kind == "MetaBlock1d":
        return _block1d(m, dtype)
    if kind == "Flat":
        return _flat
    msg = f"EfficientFormer layer {kind} has no MLX port"
    raise NotImplementedError(msg)


def efficientformer(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `EfficientFormer`: NHWC images -> normed tokens `[B, 1, N, C]`, pooled -> logits."""
    if net.global_pool != "avg" or isinstance(net.head, nn.Identity):
        msg = "EfficientFormer without the average pool and distilled heads has no MLX port"
        raise NotImplementedError(msg)
    s = net.stem
    ops: list[Callable] = [_conv(s.conv1, s.norm1, dtype), _relu, _conv(s.conv2, s.norm2, dtype), _relu]
    flat = False
    for stage in net.stages:
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_conv(stage.downsample.conv, stage.downsample.norm, dtype))
        for m in stage.blocks:
            ops.append(_layer(m, dtype))
            flat = flat or type(m).__name__ == "Flat"
    if not flat:
        ops.append(_flat)
    norm = _norm(net.norm, dtype)
    ops.append(lambda x: norm(x)[:, None])  # `[B, 1, N, C]`: the global pool averages axes 1 and 2
    with torch.no_grad():  # inference averages the two distilled heads: one linear layer
        head = SimpleNamespace(
            weight=(net.head.weight + net.head_dist.weight) / 2, bias=(net.head.bias + net.head_dist.bias) / 2
        )
    return _chain(*ops), _linear(head, dtype)
