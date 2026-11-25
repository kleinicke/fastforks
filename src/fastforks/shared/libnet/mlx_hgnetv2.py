"""HGNetV2 (timm `HighPerfGpuNet`, PP-HGNetV2 B0-B6) written in MLX, channels last.

The exported-graph translation took 0.7-8 s of the first call (export, translation, segment tuning); written out,
the first call is `mx.compile`'s trace and the shader builds only. Every batch norm folds into the convolution it
follows; the learnable affine blocks (B0-B3: one scale and shift after each ReLU) stay scalar multiply-adds, which
`mx.compile` fuses into the convolution's epilogue.
"""

# ruff: noqa: ANN401 - timm's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from fastforks._core.mlx import mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _conv, _linear

if TYPE_CHECKING:
    from collections.abc import Callable


def _lab(layer: Any) -> Callable | None:
    """`LearnableAffineBlock` -> `scale * x + bias` (Python scalars: weakly typed, the activations' dtype stays)."""
    if isinstance(layer, nn.Identity):
        return None
    scale, bias = float(layer.scale.detach()), float(layer.bias.detach())
    return lambda x: x * scale + bias


def _conv_bn_act(layer: Any, dtype: mx.Dtype) -> Callable:
    """`ConvBNAct`: conv + folded BN, then ReLU and the affine block if the layer has them."""
    ops = [_conv(layer.conv, layer.bn, dtype)]
    if not isinstance(layer.act, nn.Identity):
        ops.append(lambda x: mx.maximum(x, 0))
    if (lab := _lab(layer.lab)) is not None:
        ops.append(lab)
    return _chain(*ops)


def _layer(layer: Any, dtype: mx.Dtype) -> Callable:
    if hasattr(layer, "conv1"):  # LightConvBNAct: pointwise (no activation), then depthwise
        return _chain(_conv_bn_act(layer.conv1, dtype), _conv_bn_act(layer.conv2, dtype))
    return _conv_bn_act(layer, dtype)


def _block(block: Any, dtype: mx.Dtype) -> Callable:
    """`HighPerfGpuBlock` with the V2 ('se') aggregation: two pointwise ConvBNActs over all layers' outputs."""
    layers = [_layer(layer, dtype) for layer in block.layers]
    if type(block.aggregation[1]).__name__ == "EseModule":
        msg = "HGNet V1 (ESE aggregation) has no MLX port"
        raise NotImplementedError(msg)
    aggregate = _chain(*(_conv_bn_act(layer, dtype) for layer in block.aggregation))
    residual = block.residual

    def apply(x: mx.array) -> mx.array:
        outs = [x]
        for layer in layers:
            outs.append(layer(outs[-1]))
        y = aggregate(mx.concatenate(outs, axis=-1))
        return y + x if residual else y

    return apply


def _stem(stem: Any, dtype: mx.Dtype) -> Callable:
    """`StemV2`: the 2x2 branch and a 2x2 / stride-1 max pool of the zero-padded map, concatenated."""
    first, left, right, third, fourth = (
        _conv_bn_act(layer, dtype) for layer in (stem.stem1, stem.stem2a, stem.stem2b, stem.stem3, stem.stem4)
    )
    pad = [(0, 0), (0, 1), (0, 1), (0, 0)]

    def apply(x: mx.array) -> mx.array:
        x = mx.pad(first(x), pad)  # F.pad(x, (0, 1, 0, 1)): zeros right and bottom
        x2 = right(mx.pad(left(x), pad))
        x1 = mx.maximum(mx.maximum(x[:, :-1, :-1], x[:, 1:, :-1]), mx.maximum(x[:, :-1, 1:], x[:, 1:, 1:]))
        return fourth(third(mx.concatenate([x1, x2], axis=-1)))

    return apply


def _head(head: Any, dtype: mx.Dtype) -> Callable:
    """After the global pool: the 1x1 `last_conv` (no bias) as a matmul, ReLU, the affine block, the classifier."""
    ops: list[Callable] = []
    for layer in head.last_conv if isinstance(head.last_conv, nn.Sequential) else ():
        if isinstance(layer, nn.Conv2d):
            w = to_mx(layer.weight.detach().flatten(1).T).astype(dtype)
            mx.eval(w)
            ops.append(lambda x, w=w: x @ w)
        elif isinstance(layer, nn.ReLU):
            ops.append(lambda x: mx.maximum(x, 0))
        elif (lab := _lab(layer)) is not None:
            ops.append(lab)
    return _chain(*ops, _linear(head.fc, dtype))


def hgnetv2(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a timm `HighPerfGpuNet` with the V2 stem: NHWC images -> map, pooled `[B, C]` -> logits."""
    ops = [_stem(net.stem, dtype)]
    for stage in net.stages:
        if not isinstance(stage.downsample, nn.Identity):
            ops.append(_conv_bn_act(stage.downsample, dtype))
        ops.extend(_block(block, dtype) for block in stage.blocks)
    return _chain(*ops), _head(net.head, dtype)
