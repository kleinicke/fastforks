"""MobileNetV4 (timm `MobileNetV3` built from the MNv4 conv block defs) written in MLX, channels last.

The exported-graph translation spent about a second of the first call on export, translation and segment tuning;
written out, the first call is `mx.compile`'s trace and the shader builds only. Every `BatchNormAct2d` folds into the
convolution it follows (its activation stays an elementwise epilogue `mx.compile` fuses); the head's 1x1 conv and
batch norm after the global pool fold into one GEMM.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks._core.mlx import mx
from fastforks.shared.libnet.mlx_cnn import _chain, _linear
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv

if TYPE_CHECKING:
    from collections.abc import Callable


def _act(layer: Any) -> Callable | None:
    if isinstance(layer, nn.Identity):
        return None
    if isinstance(layer, nn.ReLU):
        return lambda x: mx.maximum(x, 0)
    if isinstance(layer, nn.ReLU6):
        return lambda x: mx.clip(x, 0, 6)
    msg = f"MobileNetV4 activation {type(layer).__name__} has no MLX port"
    raise NotImplementedError(msg)


def _conv_bn(conv: Any, bn: Any, dtype: mx.Dtype) -> Callable:
    """Conv + folded `BatchNormAct2d`, then its activation."""
    if type(conv) is not nn.Conv2d or isinstance(conv.padding, str):
        msg = "MobileNetV4 with `same` padding convolutions has no MLX port"
        raise NotImplementedError(msg)
    if not isinstance(bn.drop, nn.Identity):
        msg = "MobileNetV4 with norm dropout has no MLX port"
        raise NotImplementedError(msg)
    op, act = _conv(conv, bn, dtype), _act(bn.act)
    return op if act is None else _chain(op, act)


def _cna(layer: Any, dtype: mx.Dtype) -> Callable | None:
    """A `ConvNormAct` (or `Identity`, then None)."""
    return None if isinstance(layer, nn.Identity) else _conv_bn(layer.conv, layer.bn, dtype)


def _plain(*layers: Any) -> None:
    for layer in layers:
        if not isinstance(layer, nn.Identity):
            msg = f"MobileNetV4 block part {type(layer).__name__} has no MLX port"
            raise NotImplementedError(msg)


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    kind = type(block).__name__
    if kind == "UniversalInvertedResidual":
        _plain(block.se, block.layer_scale)
        parts = [_cna(getattr(block, n), dtype) for n in ("dw_start", "pw_exp", "dw_mid", "pw_proj", "dw_end")]
        path = _chain(*[p for p in parts if p is not None])
    elif kind == "EdgeResidual":
        _plain(block.aa, block.se)
        path = _chain(_conv_bn(block.conv_exp, block.bn1, dtype), _conv_bn(block.conv_pwl, block.bn2, dtype))
    elif kind == "ConvBnAct":
        _plain(block.aa)
        path = _conv_bn(block.conv, block.bn1, dtype)
    else:
        msg = f"MobileNetV4 block {kind} has no MLX port"
        raise NotImplementedError(msg)
    return (lambda x: x + path(x)) if block.has_skip else path


def mobilenetv4(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm MobileNetV4: NHWC images -> last map, pooled `[B, C]` -> logits."""
    if net.global_pool.pool_type != "avg" or isinstance(net.conv_head, nn.Identity):
        msg = "MobileNetV4 without the average pool and head conv has no MLX port"
        raise NotImplementedError(msg)
    blocks = [_block(b, dtype) for stage in net.blocks for b in stage]
    features = _chain(_conv_bn(net.conv_stem, net.bn1, dtype), *blocks)
    with torch.no_grad():
        head = torch.nn.utils.fuse_conv_bn_eval(net.conv_head, net.norm_head)  # the pooled 1x1 conv as a GEMM
    ops = [_linear(head, dtype)]
    if (act := _act(net.norm_head.act)) is not None:
        ops.append(act)
    _plain(net.act2)
    ops.append(_linear(net.classifier, dtype))
    return features, _chain(*ops)
