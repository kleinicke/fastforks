"""FastViT (timm `FastVit`, re-parameterized) written in MLX, channels last.

The exported-graph translation spent about a second of the first call on export, translation and segment tuning;
written out, the first call is `mx.compile`'s trace and the shader builds only. It takes the network after
`fastforks._optimize.fold.reparam`: every MobileOne / RepMixer / RepCPE block is one convolution there. The batch
norms fold into the convolution before them, the attention block's batch norm into its bias-free qkv projection and
every layer scale into the layer before it. The attention runs as MLX's fused attention, the depthwise convolutions
as `graph_mlx`'s kernel and GELU in float32, both rounded once (with float16 accumulation and a float16 GELU, t8's
logits 99th percentile |diff| was 0.071 against a 0.069 bound).
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks._core.mlx import mx
from fastforks.shared.libnet.mlx_cnn import _chain, _linear
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv
from fastforks.shared.libnet.mlx_cnn import _gelu_f32 as _gelu

if TYPE_CHECKING:
    from collections.abc import Callable


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _act(layer: Any) -> Callable | None:
    if isinstance(layer, nn.Identity):
        return None
    if isinstance(layer, nn.GELU) and layer.approximate == "none":
        return _gelu
    if isinstance(layer, nn.ReLU):
        return _relu
    msg = f"FastViT activation {type(layer).__name__} has no MLX port"
    raise NotImplementedError(msg)


def _se(se: Any, dtype: mx.Dtype) -> Callable:
    """The `SEModule` (no norm): mean, 1x1, ReLU, 1x1, sigmoid gate."""
    if not isinstance(se.bn, nn.Identity) or type(se.gate).__name__ != "Sigmoid":  # torch's or timm's own
        msg = "FastViT squeeze-excitation with a norm or another gate has no MLX port"
        raise NotImplementedError(msg)
    fc1, act, fc2 = _linear(se.fc1, dtype), _act(se.act) or (lambda x: x), _linear(se.fc2, dtype)

    def apply(x: mx.array) -> mx.array:
        s = mx.mean(x.astype(mx.float32), axis=(1, 2)).astype(x.dtype)
        return x * mx.sigmoid(fc2(act(fc1(s))))[:, None, None]

    return apply


def _rep(m: Any, dtype: mx.Dtype) -> Callable:
    """A re-parameterized `MobileOneBlock` / `ReparamLargeKernelConv`: conv, squeeze-excitation, activation."""
    if getattr(m, "reparam_conv", None) is None:
        msg = "FastViT's MLX port needs the re-parameterized network (`fastforks._optimize.fold.reparam`)"
        raise NotImplementedError(msg)
    ops = [_conv(m.reparam_conv, None, dtype)]
    if not isinstance(m.se, nn.Identity):
        ops.append(_se(m.se, dtype))
    if (act := _act(m.act)) is not None:
        ops.append(act)
    return _chain(*ops)


def _identity_plus(conv: nn.Conv2d, dtype: mx.Dtype) -> Callable:
    """A re-parameterized RepMixer / RepCPE conv as `x + delta(x)`.

    Its kernel is the identity plus a small update, whose centre taps (about 1) keep float16's 1e-3 resolution only for
    the update's own digits once split off (folded, t8's logits mean |diff| 0.029 against a 0.023 bound).
    """
    with torch.no_grad():
        delta = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, groups=conv.groups
        )
        weight = conv.weight.detach().clone()
        kh, kw = weight.shape[-2:]
        per_group = weight.shape[1]
        rows = torch.arange(weight.shape[0])
        weight[rows, rows % per_group, kh // 2, kw // 2] -= 1
        delta.weight.copy_(weight)
        delta.bias.copy_(conv.bias.detach())
    apply = _conv(delta, None, dtype)
    return lambda x: x + apply(x)


def _scale(layer: Any) -> torch.Tensor:
    return layer.gamma.detach().flatten()


def _mlp(m: Any, gamma: torch.Tensor, dtype: mx.Dtype) -> Callable:
    """`ConvMlp`: depthwise 7x7 + BN, 1x1, GELU, 1x1 (the layer scale folded in)."""
    if not isinstance(m.conv.bn.act, nn.Identity) or not isinstance(m.conv.bn.drop, nn.Identity):
        msg = "FastViT MLP with an activated norm has no MLX port"
        raise NotImplementedError(msg)
    act = _act(m.act)
    ops = [_conv(m.conv.conv, m.conv.bn, dtype), _linear(m.fc1, dtype), *([act] if act else [])]
    return _chain(*ops, _linear(m.fc2, dtype, gamma))


def _attention(block: Any, dtype: mx.Dtype) -> Callable:
    """`AttentionBlock`'s mixer: BN (folded into qkv), multi-head attention on `[B, N, C]` tokens, proj."""
    attn, bn = block.token_mixer, block.norm
    if attn.qkv.bias is not None:
        msg = "FastViT attention with a qkv bias has no MLX port"
        raise NotImplementedError(msg)
    with torch.no_grad():  # qkv(x * s + t) = x @ (s * W^T) + t @ W^T
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        t = bn.bias - bn.running_mean * s
        w = attn.qkv.weight
        qkv = _linear(SimpleNamespace(weight=w * s[None], bias=w @ t), dtype)
    proj = _linear(attn.proj, dtype, _scale(block.layer_scale_1))
    heads, hd, scale = attn.num_heads, attn.head_dim, attn.scale

    def apply(x: mx.array) -> mx.array:
        b, h, w_, c = x.shape
        n = h * w_
        t = qkv(x.reshape(b, n, c)).reshape(b, n, 3, heads, hd).transpose(2, 0, 3, 1, 4)
        o = mx.fast.scaled_dot_product_attention(t[0], t[1], t[2], scale=scale)
        return proj(o.transpose(0, 2, 1, 3).reshape(b, n, c)).reshape(b, h, w_, c)

    return apply


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    kind = type(block).__name__
    if kind == "RepMixerBlock":
        if block.token_mixer.reparam_conv is None:
            msg = "FastViT's MLX port needs the re-parameterized network (`fastforks._optimize.fold.reparam`)"
            raise NotImplementedError(msg)
        mixer = _identity_plus(block.token_mixer.reparam_conv, dtype)
        mlp = _mlp(block.mlp, _scale(block.layer_scale), dtype)

        def repmixer(x: mx.array) -> mx.array:
            x = mixer(x)
            return x + mlp(x)

        return repmixer
    if kind == "AttentionBlock":
        attn = _attention(block, dtype)
        mlp = _mlp(block.mlp, _scale(block.layer_scale_2), dtype)

        def attention(x: mx.array) -> mx.array:
            x = x + attn(x)
            return x + mlp(x)

        return attention
    msg = f"FastViT block {kind} has no MLX port"
    raise NotImplementedError(msg)


def fastvit(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a re-parameterized timm `FastVit`: NHWC images -> final conv map, pooled -> logits."""
    head = net.head
    if getattr(net, "fork_feat", False) or head.global_pool.pool_type != "avg":
        msg = "FastViT without the average-pool classifier head has no MLX port"
        raise NotImplementedError(msg)
    ops = [_rep(m, dtype) for m in net.stem]
    for stage in net.stages:
        if not isinstance(stage.downsample, nn.Identity):
            ops.extend(_rep(m, dtype) for m in stage.downsample.proj)
        if not isinstance(stage.pos_emb, nn.Identity):
            if stage.pos_emb.reparam_conv is None:
                msg = "FastViT's MLX port needs the re-parameterized network (`fastforks._optimize.fold.reparam`)"
                raise NotImplementedError(msg)
            ops.append(_identity_plus(stage.pos_emb.reparam_conv, dtype))
        ops.extend(_block(b, dtype) for b in stage.blocks)
    ops.append(_rep(net.final_conv, dtype))
    return _chain(*ops), _linear(head.fc, dtype)
