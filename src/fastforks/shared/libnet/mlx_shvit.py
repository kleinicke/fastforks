"""SHViT (timm `SHViT`, S1-S4) written in MLX, channels last.

The exported-graph translation spent about a second of the first call on export, translation and segment tuning;
written out, the first call is `mx.compile`'s trace and the shader builds only. Every `Conv2dNorm` folds its batch
norm into the convolution and the head's `NormLinear` folds into one linear layer (timm's own `fuse`). The
single-head attention runs on the first `pdim` channels only, as MLX's fused attention with one head.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from fastforks._core.mlx import mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _linear
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv

if TYPE_CHECKING:
    from collections.abc import Callable


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _cn(layer: Any, dtype: mx.Dtype) -> Callable:
    """`Conv2dNorm`: conv (no bias) + folded batch norm."""
    return _conv(layer.c, layer.bn, dtype)


def _se(se: Any, dtype: mx.Dtype) -> Callable:
    if not isinstance(se.bn, nn.Identity) or getattr(se, "add_maxpool", False):
        msg = "SHViT squeeze-excitation with a norm or max pool has no MLX port"
        raise NotImplementedError(msg)
    fc1, fc2 = _linear(se.fc1, dtype), _linear(se.fc2, dtype)

    def apply(x: mx.array) -> mx.array:
        s = mx.mean(x.astype(mx.float32), axis=(1, 2)).astype(x.dtype)
        return x * mx.sigmoid(fc2(_relu(fc1(s))))[:, None, None]

    return apply


def _merge(m: Any, dtype: mx.Dtype) -> Callable:
    """`PatchMerging`: 1x1, ReLU, depthwise 3x3 (stride 2), ReLU, squeeze-excitation, 1x1."""
    se = None if isinstance(m.se, nn.Identity) else _se(m.se, dtype)
    ops = [_cn(m.conv1, dtype), _relu, _cn(m.conv2, dtype), _relu, *([se] if se else []), _cn(m.conv3, dtype)]
    return _chain(*ops)


def _group_norm1(layer: Any, dtype: mx.Dtype) -> Callable:
    """`GroupNorm1`: one group, so the statistics run over all channels and pixels of an image (float32)."""
    w, b, eps = to_mx(layer.weight.detach()), to_mx(layer.bias.detach()), layer.eps
    mx.eval(w, b)

    def apply(x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        mean = mx.mean(y, axis=(1, 2, 3), keepdims=True)
        var = mx.var(y, axis=(1, 2, 3), keepdims=True)
        return ((y - mean) * mx.rsqrt(var + eps) * w + b).astype(dtype)

    return apply


def _shsa(m: Any, dtype: mx.Dtype) -> Callable:
    norm, qkv, proj = _group_norm1(m.pre_norm, dtype), _cn(m.qkv, dtype), _cn(m.proj[1], dtype)
    if not isinstance(m.proj[0], nn.ReLU):
        msg = "SHViT attention without the ReLU projection has no MLX port"
        raise NotImplementedError(msg)
    pdim, qk, scale = m.pdim, m.qk_dim, m.scale

    def apply(x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        t = qkv(norm(x[..., :pdim])).reshape(b, 1, h * w, 2 * qk + pdim)
        q, k, v = t[..., :qk], t[..., qk : 2 * qk], t[..., 2 * qk :]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).reshape(b, h, w, pdim)
        return proj(_relu(mx.concatenate([o, x[..., pdim:]], axis=-1)))

    return apply


def _ffn(m: Any, dtype: mx.Dtype) -> Callable:
    return _chain(_cn(m.pw1, dtype), _relu, _cn(m.pw2, dtype))


def _layer(m: Any, dtype: mx.Dtype) -> Callable | None:
    """A `Residual`, `PatchMerging` or `BasicBlock` (`Identity`: None)."""
    kind = type(m).__name__
    if kind == "Identity":
        return None
    if kind == "Residual":
        inner = type(m.m).__name__
        path = {"Conv2dNorm": _cn, "FFN": _ffn, "SHSA": _shsa}[inner](m.m, dtype)
        return lambda x: x + path(x)
    if kind == "PatchMerging":
        return _merge(m, dtype)
    if kind == "BasicBlock":
        return _chain(*[op for op in (_layer(m.conv, dtype), _layer(m.mixer, dtype), _layer(m.ffn, dtype)) if op])
    msg = f"SHViT layer {kind} has no MLX port"
    raise NotImplementedError(msg)


def shvit(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `SHViT`: NHWC images -> last map, pooled `[B, C]` -> logits."""
    if net.global_pool.pool_type != "avg" or type(net.head).__name__ != "NormLinear":
        msg = "SHViT without the average pool and `NormLinear` head has no MLX port"
        raise NotImplementedError(msg)
    ops: list[Callable] = [_relu if isinstance(layer, nn.ReLU) else _cn(layer, dtype) for layer in net.patch_embed]
    for stage in net.stages:
        downsample = list(stage.downsample) if isinstance(stage.downsample, nn.Sequential) else [stage.downsample]
        ops.extend(op for m in [*downsample, *stage.blocks] if (op := _layer(m, dtype)) is not None)
    return _chain(*ops), _linear(net.head.fuse(), dtype)
