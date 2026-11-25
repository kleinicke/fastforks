"""CoAtNet (timm `MaxxVit` with MBConv and full-map transformer stages) written in MLX, channels last.

The exported-graph translation took 0.8-2.9 s of the first call (export, translation, segment tuning); written out,
the first call is `mx.compile`'s trace and the shader builds only. Batch norms fold into the convolutions they
follow (a pre-norm before a pool stays an affine); the transformer stages' relative-position bias (`RelPosBias`'s
table or `RelPosMlp`'s MLP) is a constant of the model's input size, computed once in PyTorch.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks._core.mlx import mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _activation, _chain, _conv, _linear, _norm, _pool2d

if TYPE_CHECKING:
    from collections.abc import Callable


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU computed in float32 (`1 + erf` of a negative input cancels in float16)."""
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


def _act(layer: nn.Module) -> Callable:
    if isinstance(layer, (nn.Identity, nn.Dropout)):
        return lambda x: x
    if type(layer).__name__ == "GELU" and getattr(layer, "approximate", "none") == "none":  # timm's or torch's
        return _gelu
    if type(layer).__name__ == "Sigmoid":  # timm's
        return mx.sigmoid
    return _activation(layer)


def _bn_act(layer: Any, dtype: mx.Dtype) -> Callable:
    """A batch norm on its own (timm's `BatchNormAct2d` or a plain `BatchNorm2d`): the affine, then any activation."""
    scale = layer.weight.detach().float() / (layer.running_var.float() + layer.eps).sqrt()
    gain = to_mx(scale).astype(dtype)
    shift = to_mx(layer.bias.detach().float() - layer.running_mean.float() * scale).astype(dtype)
    mx.eval(gain, shift)
    act = _act(getattr(layer, "act", nn.Identity()))
    return lambda x: act(x * gain + shift)


def _conv_bn_act(conv: nn.Conv2d, bn: Any, dtype: mx.Dtype) -> Callable:
    return _chain(_conv(conv, bn, dtype), _act(bn.act))


def _op(layer: Any, dtype: mx.Dtype) -> Callable:  # noqa: PLR0911 - one return per layer kind
    """The simple layers between the blocks: identities, pools, 1x1 expansions, norms."""
    name = type(layer).__name__
    if isinstance(layer, (nn.Identity, nn.Dropout)):
        return lambda x: x
    if isinstance(layer, nn.Sequential):
        return _chain(*(_op(m, dtype) for m in layer))
    if isinstance(layer, nn.Conv2d):
        return _conv(layer, None, dtype)
    if isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
        if isinstance(layer, nn.AvgPool2d) and layer.padding and not layer.count_include_pad:
            msg = "CoAtNet on MLX: padded average pooling without the padding in the count"
            raise ValueError(msg)
        return _pool2d(layer)
    if name == "Downsample2d":
        return _chain(_op(layer.pool, dtype), _op(layer.expand, dtype))
    if name == "LayerNorm2d":
        return _norm(layer, dtype)
    if name == "BatchNormAct2d":
        return _bn_act(layer, dtype)
    return _act(layer)


def _se(layer: Any, dtype: mx.Dtype) -> Callable:
    if getattr(layer, "add_maxpool", False) or not isinstance(layer.bn, nn.Identity):
        msg = "CoAtNet on MLX: SE with max pooling or a norm"
        raise ValueError(msg)
    gate = _chain(_linear(layer.fc1, dtype), _act(layer.act), _linear(layer.fc2, dtype), _act(layer.gate))

    def apply(x: mx.array) -> mx.array:
        pooled = mx.mean(x.astype(mx.float32), axis=(1, 2), keepdims=True).astype(x.dtype)
        return x * gate(pooled)

    return apply


def _depthwise(conv: nn.Conv2d, bn: Any, dtype: mx.Dtype) -> Callable:
    """The depthwise conv, its norm and the norm's activation; in float16, the norm in float32.

    The in12k CoAtNets scale the depthwise output up to 2e2-4e3 (`rmlp_2_rw`: activations of 5e3 in, the norm's
    output 1.4e4 out): folded into a float16 conv, its products overflow. The conv itself (outputs under 1e3) stays
    float16, and the norm's affine runs in float32 before the cast back (as an unfolded PyTorch batch norm does).
    """
    if dtype != mx.float16:
        return _conv_bn_act(conv, bn, dtype)
    raw, norm = _conv(conv, None, dtype), _bn_act(bn, mx.float32)
    return lambda x: norm(raw(x).astype(mx.float32)).astype(x.dtype)


def _mbconv(block: Any, dtype: mx.Dtype) -> Callable:
    shortcut = _op(block.shortcut, dtype)
    ops = [_op(block.pre_norm, dtype), _op(block.down, dtype), _conv_bn_act(block.conv1_1x1, block.norm1, dtype)]
    if block.se_early is not None:  # SE between the depthwise conv and its norm: no fold
        ops += [_conv(block.conv2_kxk, None, dtype), _se(block.se_early, dtype), _bn_act(block.norm2, dtype)]
    else:
        ops.append(_depthwise(block.conv2_kxk, block.norm2, dtype))
    if block.se is not None:
        ops.append(_se(block.se, dtype))
    ops.append(_conv(block.conv3_1x1, None, dtype))
    path = _chain(*ops)
    return lambda x: path(x) + shortcut(x)


def _gamma(layer: Any) -> torch.Tensor | None:
    """A `LayerScale2d`'s scale (folded into the projection before it), or None for an identity."""
    if isinstance(layer, nn.Identity):
        return None
    return layer.gamma.detach().float()


def _attention(attn: Any, dtype: mx.Dtype, gamma: torch.Tensor | None) -> Callable:
    """`Attention2d` over the whole map with its relative-position bias, on `[B, H, W, C]`."""
    if not attn.head_first or attn.rel_pos is None:
        msg = "CoAtNet on MLX: attention without heads first or without its own relative-position bias"
        raise ValueError(msg)
    heads, d, scale = attn.num_heads, attn.dim_head, attn.scale
    qkv, proj = _linear(attn.qkv, dtype), _linear(attn.proj, dtype, gamma)
    with torch.no_grad():
        bias = to_mx(attn.rel_pos.get_bias().float()[0]).astype(dtype)  # [heads, N, N] at the model's input size
    mx.eval(bias)

    def apply(x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        # the qkv channels are (head, q / k / v, dim) (`head_first`)
        t = qkv(x.reshape(b, h * w, -1)).reshape(b, h * w, heads, 3, d).transpose(3, 0, 2, 1, 4)
        out = mx.fast.scaled_dot_product_attention(t[0], t[1], t[2], scale=scale, mask=bias)
        return proj(out.transpose(0, 2, 1, 3).reshape(b, h, w, heads * d))

    return apply


def _transformer(block: Any, dtype: mx.Dtype) -> Callable:
    shortcut, norm1, norm2 = _op(block.shortcut, dtype), _op(block.norm1, dtype), _op(block.norm2, dtype)
    attn = _attention(block.attn, dtype, _gamma(block.ls1))
    mlp = block.mlp
    if not isinstance(mlp.norm, nn.Identity):
        msg = "CoAtNet on MLX: MLP with a norm"
        raise TypeError(msg)
    ffn = _chain(_linear(mlp.fc1, dtype), _act(mlp.act), _linear(mlp.fc2, dtype, _gamma(block.ls2)))

    def apply(x: mx.array) -> mx.array:
        x = shortcut(x) + attn(norm1(x))
        return x + ffn(norm2(x))

    return apply


def coatnet(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm CoAtNet: NHWC image -> normalised NHWC features; pooled `[B, C]` -> logits."""
    stem = net.stem
    ops = [_conv_bn_act(stem.conv1, stem.norm1, dtype), _conv(stem.conv2, None, dtype)]
    for stage in net.stages:
        for block in stage.blocks:
            kind = type(block).__name__
            if kind == "MbConvBlock":
                ops.append(_mbconv(block, dtype))
            elif kind == "TransformerBlock2d":
                ops.append(_transformer(block, dtype))
            else:
                msg = f"CoAtNet on MLX: no {kind} block"
                raise ValueError(msg)
    ops.append(_op(net.norm, dtype))
    head = net.head
    if head.global_pool.pool_type != "avg" or not isinstance(head.flatten, nn.Identity):
        msg = "CoAtNet on MLX: a head other than average pool + linear"
        raise ValueError(msg)
    return _chain(*ops), _linear(head.fc, dtype)
