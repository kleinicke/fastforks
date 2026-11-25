"""EfficientFormerV2 (timm) written in MLX, channels last.

The exported-graph translation took 1.4-2.3 s of the first call; written out, the first call is `mx.compile`'s trace and
the shader builds only. Batch norms fold into the convolutions they follow and the layer scales into the projections
before them; the attention biases (a table gathered by relative position) are constants of the model's input size,
gathered once in PyTorch. The talking-head 1x1 convolutions mix the heads, so they run as a matrix product over a
heads-last view, the first one with the scale and the bias folded in.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks._core.mlx import mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _conv, _linear
from fastforks.shared.libnet.mlx_coatnet import _act, _bn_act, _conv_bn_act, _gamma
from fastforks.shared.mlx_seg import resize

if TYPE_CHECKING:
    from collections.abc import Callable


def _conv_norm(layer: Any, dtype: mx.Dtype, gamma: torch.Tensor | None = None) -> Callable:
    """A `ConvNorm` (conv + batch norm, no activation), scaled by a following `LayerScale2d`'s `gamma`."""
    conv = torch.nn.utils.fuse_conv_bn_eval(layer.conv, layer.bn)
    if gamma is not None:
        with torch.no_grad():
            conv.weight.mul_(gamma[:, None, None, None])
            conv.bias.mul_(gamma)
    return _conv(conv, None, dtype)


def _heads(x: mx.array, heads: int) -> mx.array:
    """`[B, H, W, heads * d]` (channel = head * d + i) -> `[B, heads, H * W, d]`."""
    b, h, w, c = x.shape
    return x.reshape(b, h * w, heads, c // heads).transpose(0, 2, 1, 3)


def _merge(x: mx.array, h: int, w: int) -> mx.array:
    """`[B, heads, N, d]` -> `[B, H, W, heads * d]`."""
    b, heads, _, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, h, w, heads * d)


def _bias(attn: Any, dtype: mx.Dtype) -> mx.array:
    with torch.no_grad():
        bias = to_mx(attn.attention_biases.detach().float()[:, attn.attention_bias_idxs]).astype(dtype)
    mx.eval(bias)
    return bias  # [heads, N_q, N_k]


def _attention(attn: Any, dtype: mx.Dtype, gamma: torch.Tensor | None) -> Callable:
    """`Attention2d`: full-map attention with talking heads, at stride 2 in stage 3 (then upsampled back)."""
    heads, scale = attn.num_heads, attn.scale
    stride = _conv_norm(attn.stride_conv, dtype) if attn.stride_conv is not None else None
    q, k, v = (_conv_norm(m, dtype) for m in (attn.q, attn.k, attn.v))
    v_local = _conv_norm(attn.v_local, dtype)
    act, proj = _act(attn.act), _conv_norm(attn.proj, dtype, gamma)
    with torch.no_grad():
        w1 = attn.talking_head1.weight.detach().float()[:, :, 0, 0]
        b1 = attn.talking_head1.bias.detach().float()
        w2 = attn.talking_head2.weight.detach().float()[:, :, 0, 0]
        b2 = attn.talking_head2.bias.detach().float()
        table = attn.attention_biases.detach().float()[:, attn.attention_bias_idxs]  # [heads, N, N]
        # talking_head1(scale * qk + table) = (scale * W1) qk + (W1 table + b1), all heads-last
        mix1 = to_mx((w1 * scale).T.contiguous()).astype(dtype)
        bias1 = to_mx((torch.einsum("gh,hnm->nmg", w1, table) + b1).contiguous()).astype(dtype)
        mix2, bias2 = to_mx(w2.T.contiguous()).astype(dtype), to_mx(b2).astype(dtype)
    mx.eval(mix1, bias1, mix2, bias2)
    upsample = attn.upsample is not None

    def apply(x: mx.array) -> mx.array:
        size = x.shape[1:3]
        if stride is not None:
            x = stride(x)
        _, h, w, _ = x.shape
        vv = v(x)
        local = v_local(vv)
        logits = _heads(q(x), heads) @ _heads(k(x), heads).transpose(0, 1, 3, 2)  # [B, heads, N, N]
        mixed = logits.transpose(0, 2, 3, 1) @ mix1 + bias1  # [B, N, N, heads]
        weights = mx.softmax(mixed, axis=2, precise=True) @ mix2 + bias2
        out = _merge(weights.transpose(0, 3, 1, 2) @ _heads(vv, heads), h, w) + local
        if upsample:
            out = resize(out, size)
        return proj(act(out))

    return apply


def _attention_down(attn: Any, dtype: mx.Dtype) -> Callable:
    """`Attention2dDownsample`: stride-2 queries (a depthwise conv plus a subsampling) over the full-map keys."""
    heads, scale = attn.num_heads, attn.scale
    local_q, q_proj = _conv(attn.q.local, None, dtype), _conv_norm(attn.q.proj, dtype)
    k, v, v_local = _conv_norm(attn.k, dtype), _conv_norm(attn.v, dtype), _conv_norm(attn.v_local, dtype)
    act, proj = _act(attn.act), _conv_norm(attn.proj, dtype)
    bias = _bias(attn, dtype)

    def apply(x: mx.array) -> mx.array:
        vv = v(x)
        local = v_local(vv)
        query = q_proj(local_q(x) + x[:, ::2, ::2])  # `AvgPool2d(1, 2)` is a subsampling
        _, h, w, _ = query.shape
        logits = (_heads(query, heads) * scale) @ _heads(k(x), heads).transpose(0, 1, 3, 2) + bias
        weights = mx.softmax(logits, axis=-1, precise=True)
        return proj(act(_merge(weights @ _heads(vv, heads), h, w) + local))

    return apply


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    mlp = block.mlp
    ffn = _chain(
        _conv_bn_act(mlp.fc1.conv, mlp.fc1.bn, dtype),
        _conv_bn_act(mlp.mid.conv, mlp.mid.bn, dtype),
        _conv_norm(mlp.fc2, dtype, _gamma(block.ls2)),
    )
    if block.token_mixer is None:
        return lambda x: x + ffn(x)
    mixer = _attention(block.token_mixer, dtype, _gamma(block.ls1))

    def apply(x: mx.array) -> mx.array:
        x = x + mixer(x)
        return x + ffn(x)

    return apply


def _downsample(layer: Any, dtype: mx.Dtype) -> Callable:
    conv = _conv_norm(layer.conv, dtype)
    if layer.attn is None:
        return conv
    attn = _attention_down(layer.attn, dtype)
    return lambda x: attn(x) + conv(x)


def efficientformerv2(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm EfficientFormerV2: NHWC image -> normalised features; pooled -> averaged logits."""
    if net.global_pool != "avg" or net.head_dist is None:
        msg = "EfficientFormerV2 on MLX: a head other than average pool + the two averaged linear heads"
        raise ValueError(msg)
    ops = [_conv_bn_act(m.conv, m.bn, dtype) for m in (net.stem.conv1, net.stem.conv2)]
    for stage in net.stages:
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_downsample(stage.downsample, dtype))
        ops.extend(_block(block, dtype) for block in stage.blocks)
    ops.append(_bn_act(net.norm, dtype))
    head = nn.Linear(net.head.in_features, net.head.out_features)
    with torch.no_grad():  # eval averages the class and distillation heads
        head.weight.copy_((net.head.weight + net.head_dist.weight) / 2)
        head.bias.copy_((net.head.bias + net.head_dist.bias) / 2)
    return _chain(*ops), _linear(head, dtype)
