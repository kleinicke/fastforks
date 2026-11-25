"""EfficientViT (MIT, timm `EfficientVit` B-series) written in MLX, channels last.

The exported-graph translation spent about a second of the first call on export, translation and segment tuning;
written out, the first call is `mx.compile`'s trace and the shader builds only. Every batch norm folds into the
convolution it follows. The ReLU linear attention (`LiteMLA`) runs in float32 as upstream (it disables autocast).
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from fastforks._core.mlx import mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _norm
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv

if TYPE_CHECKING:
    from collections.abc import Callable


def _hardswish(x: mx.array) -> mx.array:
    return x * mx.clip(x + 3, 0, 6) * (1 / 6)


def _act(layer: Any) -> Callable | None:
    if isinstance(layer, nn.Identity):
        return None
    if isinstance(layer, nn.Hardswish):
        return _hardswish
    if isinstance(layer, nn.ReLU):
        return lambda x: mx.maximum(x, 0)
    msg = f"EfficientViT activation {type(layer).__name__} has no MLX port"
    raise NotImplementedError(msg)


def _cna(layer: Any, dtype: mx.Dtype) -> Callable:
    """`ConvNormAct`: conv + folded batch norm (if any), then the activation."""
    bn = None if isinstance(layer.norm, nn.Identity) else layer.norm
    op, act = _conv(layer.conv, bn, dtype), _act(layer.act)
    return op if act is None else _chain(op, act)


def _dense(layer: nn.Linear, dtype: mx.Dtype) -> Callable:
    w = to_mx(layer.weight.detach().T).astype(dtype)
    b = None if layer.bias is None else to_mx(layer.bias.detach()).astype(dtype)
    mx.eval(w, *([] if b is None else [b]))
    return (lambda x: x @ w) if b is None else (lambda x: mx.addmm(b, x, w))


def _lite_mla(m: Any, dtype: mx.Dtype) -> Callable:
    qkv = _cna(m.qkv, dtype)
    aggreg = [_chain(_conv(op[0], None, dtype), _conv(op[1], None, dtype)) for op in m.aggreg]
    if not isinstance(m.kernel_func, nn.ReLU):
        msg = "EfficientViT attention without the ReLU kernel has no MLX port"
        raise NotImplementedError(msg)
    proj, dim, eps = _cna(m.proj, dtype), m.dim, m.eps

    def apply(x: mx.array) -> mx.array:
        b, h, w, _ = x.shape
        t = qkv(x)
        t = mx.concatenate([t, *[op(t) for op in aggreg]], axis=-1) if aggreg else t
        t = t.reshape(b, h * w, -1, 3 * dim).transpose(0, 2, 1, 3).astype(mx.float32)  # [B, heads, N, 3 * dim]
        q, k, v = mx.maximum(t[..., :dim], 0), mx.maximum(t[..., dim : 2 * dim], 0), t[..., 2 * dim :]
        kv = k.swapaxes(-1, -2) @ v  # [B, heads, dim, dim]
        ks = mx.sum(k, axis=-2, keepdims=True)  # the padded ones column of v: sum over tokens of k
        out = (q @ kv) / ((q * ks).sum(-1, keepdims=True) + eps)
        out = out.astype(x.dtype).transpose(0, 2, 1, 3).reshape(b, h, w, -1)
        return proj(out)

    return apply


def _main(m: Any, dtype: mx.Dtype) -> Callable:
    kind = type(m).__name__
    if kind == "DSConv":
        return _chain(_cna(m.depth_conv, dtype), _cna(m.point_conv, dtype))
    if kind == "MBConv":
        return _chain(_cna(m.inverted_conv, dtype), _cna(m.depth_conv, dtype), _cna(m.point_conv, dtype))
    if kind == "LiteMLA":
        return _lite_mla(m, dtype)
    msg = f"EfficientViT block {kind} has no MLX port"
    raise NotImplementedError(msg)


def _residual(block: Any, dtype: mx.Dtype) -> Callable:
    if not isinstance(block.pre_norm, nn.Identity):
        msg = "EfficientViT pre-norm blocks have no MLX port"
        raise NotImplementedError(msg)
    main = _main(block.main, dtype)
    if block.shortcut is None:
        return main
    if not isinstance(block.shortcut, nn.Identity):
        msg = "EfficientViT projected shortcuts have no MLX port"
        raise NotImplementedError(msg)
    return lambda x: x + main(x)


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    if type(block).__name__ == "EfficientVitBlock":
        return _chain(_residual(block.context_module, dtype), _residual(block.local_module, dtype))
    return _residual(block, dtype)


def efficientvit(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm MIT `EfficientVit`: NHWC images -> head conv map, pooled `[B, C]` -> logits."""
    head = net.head
    if type(head).__name__ != "ClassifierHead" or head.global_pool.pool_type != "avg":
        msg = "EfficientViT without the average-pool classifier head has no MLX port"
        raise NotImplementedError(msg)
    stem = [_cna(net.stem.in_conv, dtype)]
    stem += [_block(b, dtype) for name, b in net.stem.named_children() if name != "in_conv"]
    blocks = [_block(b, dtype) for stage in net.stages for b in stage.blocks]
    features = _chain(*stem, *blocks, _cna(head.in_conv, dtype))
    classifier: list[Callable] = []
    for layer in head.classifier:
        if isinstance(layer, nn.Linear):
            classifier.append(_dense(layer, dtype))
        elif isinstance(layer, nn.LayerNorm):
            classifier.append(_norm(layer, dtype))
        elif not isinstance(layer, nn.Dropout) and (act := _act(layer)) is not None:
            classifier.append(act)
    return features, _chain(*classifier)
