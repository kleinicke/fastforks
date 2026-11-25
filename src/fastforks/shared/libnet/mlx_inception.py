"""Inception v3 (torchvision `Inception3`) written in MLX, channels last.

The exported-graph translation spent about a second of the first call on export, translation and segment tuning;
written out, the first call is `mx.compile`'s trace and the shader builds only. Every `BasicConv2d` folds its batch
norm into the convolution. In each mixed block the 1x1 convolutions that read the block input run as one GEMM, the
pool branch's 1x1 among them: its bias-free 1x1 runs before the 3x3 average pool (both linear, the pool counts the
zero padding), so the pool reads 32-192 channels instead of 192-2048.
"""

# ruff: noqa: ANN401 - torchvision's architecture-specific layers

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.mlx import mx, to_mx
from fastforks.shared.libnet.mlx_cnn import _chain, _linear
from fastforks.shared.libnet.mlx_cnn import _conv_k as _conv
from fastforks.shared.mlx_graph import conv1x1

if TYPE_CHECKING:
    from collections.abc import Callable


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _basic(layer: Any, dtype: mx.Dtype) -> Callable:
    """`BasicConv2d`: conv + folded batch norm + ReLU."""
    return _chain(_conv(layer.conv, layer.bn, dtype), _relu)


def _pointwise(layers: list[Any], pooled: Any, dtype: mx.Dtype) -> Callable:
    """The block's 1x1 `BasicConv2d`s (and the pool branch's, without its bias) as one GEMM -> their outputs."""
    with torch.no_grad():
        folded = [torch.nn.utils.fuse_conv_bn_eval(m.conv, m.bn) for m in [*layers, pooled]]
        weight = torch.cat([f.weight[:, :, 0, 0] for f in folded]).t().contiguous()
        bias = torch.cat([f.bias for f in folded[:-1]] + [torch.zeros_like(folded[-1].bias)])
    w, b, pool_bias = to_mx(weight).astype(dtype), to_mx(bias).astype(dtype), to_mx(folded[-1].bias).astype(dtype)
    mx.eval(w, b, pool_bias)
    cuts = [int(c) for c in torch.tensor([f.out_channels for f in folded]).cumsum(0)[:-1]]

    def apply(x: mx.array) -> list[mx.array]:
        *parts, pre = mx.split(conv1x1(x, w, b), cuts, axis=-1)
        return [_relu(p) for p in parts] + [_relu(_avg3(pre) + pool_bias)]

    return apply


def _avg3(x: mx.array) -> mx.array:
    """3x3 average pool, stride 1, zero padding 1 counted (`F.avg_pool2d`'s default), as separable sums."""
    h, w = x.shape[1], x.shape[2]
    x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    x = x[:, :h] + x[:, 1 : h + 1] + x[:, 2 : h + 2]
    return (x[:, :, :w] + x[:, :, 1 : w + 1] + x[:, :, 2 : w + 2]) * (1 / 9)


def _max3(x: mx.array) -> mx.array:
    """3x3 max pool, stride 2, no padding."""
    oh, ow = (x.shape[1] - 3) // 2 + 1, (x.shape[2] - 3) // 2 + 1
    taps = [x[:, i : i + 2 * oh - 1 : 2, j : j + 2 * ow - 1 : 2] for i in range(3) for j in range(3)]
    out = taps[0]
    for t in taps[1:]:
        out = mx.maximum(out, t)
    return out


def _split_pair(first: Callable, a: Callable, b: Callable) -> Callable:
    """`cat(a(y), b(y))` of `y = first(x)` (Inception E's split 1x3 / 3x1 branches)."""

    def apply(x: mx.array) -> mx.array:
        y = first(x)
        return mx.concatenate([a(y), b(y)], axis=-1)

    return apply


def _mixed(block: Any, dtype: mx.Dtype) -> Callable:
    kind = type(block).__name__
    b = lambda name: _basic(getattr(block, name), dtype)  # noqa: E731
    if kind == "InceptionA":
        pw = _pointwise([block.branch1x1, block.branch5x5_1, block.branch3x3dbl_1], block.branch_pool, dtype)
        tails = [None, b("branch5x5_2"), _chain(b("branch3x3dbl_2"), b("branch3x3dbl_3")), None]
    elif kind == "InceptionC":
        names = ("branch1x1", "branch7x7_1", "branch7x7dbl_1")
        pw = _pointwise([getattr(block, n) for n in names], block.branch_pool, dtype)
        dbl = _chain(*[b(f"branch7x7dbl_{i}") for i in range(2, 6)])
        tails = [None, _chain(b("branch7x7_2"), b("branch7x7_3")), dbl, None]
    elif kind == "InceptionE":
        pw = _pointwise([block.branch1x1, block.branch3x3_1, block.branch3x3dbl_1], block.branch_pool, dtype)
        tails = [
            None,
            _split_pair(lambda y: y, b("branch3x3_2a"), b("branch3x3_2b")),
            _split_pair(b("branch3x3dbl_2"), b("branch3x3dbl_3a"), b("branch3x3dbl_3b")),
            None,
        ]
    elif kind == "InceptionB":
        single, dbl1 = b("branch3x3"), b("branch3x3dbl_1")
        dbl = _chain(b("branch3x3dbl_2"), b("branch3x3dbl_3"))
        return lambda x: mx.concatenate([single(x), dbl(dbl1(x)), _max3(x)], axis=-1)
    elif kind == "InceptionD":
        pw2 = _pointwise_pair(block.branch3x3_1, block.branch7x7x3_1, dtype)
        t3 = b("branch3x3_2")
        t7 = _chain(*[b(f"branch7x7x3_{i}") for i in range(2, 5)])

        def reduce(x: mx.array) -> mx.array:
            y3, y7 = pw2(x)
            return mx.concatenate([t3(y3), t7(y7), _max3(x)], axis=-1)

        return reduce
    else:
        msg = f"Inception block {kind} has no MLX port"
        raise NotImplementedError(msg)

    def apply(x: mx.array) -> mx.array:
        parts = pw(x)
        return mx.concatenate([p if t is None else t(p) for p, t in zip(parts, tails, strict=True)], axis=-1)

    return apply


def _pointwise_pair(first: Any, second: Any, dtype: mx.Dtype) -> Callable:
    """Two 1x1 `BasicConv2d`s of one input as one GEMM."""
    with torch.no_grad():
        folded = [torch.nn.utils.fuse_conv_bn_eval(m.conv, m.bn) for m in (first, second)]
        w = to_mx(torch.cat([f.weight[:, :, 0, 0] for f in folded]).t().contiguous()).astype(dtype)
        b = to_mx(torch.cat([f.bias for f in folded])).astype(dtype)
    mx.eval(w, b)
    cut = folded[0].out_channels

    def apply(x: mx.array) -> list[mx.array]:
        return mx.split(_relu(conv1x1(x, w, b)), [cut], axis=-1)

    return apply


def inception(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    """`(features, head)` of a torchvision `Inception3`: NHWC images -> last map, pooled `[B, C]` -> logits."""
    ops: list[Callable] = []
    if net.transform_input:  # the ImageNet normalisation of the input, remapped to (x - 0.5) / 0.5
        scale = mx.array([0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5], dtype)
        shift = mx.array([(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5], dtype)
        ops.append(lambda x: x * scale + shift)
    stem = ("Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", None, "Conv2d_3b_1x1", "Conv2d_4a_3x3", None)
    ops.extend(_max3 if name is None else _basic(getattr(net, name), dtype) for name in stem)
    mixed = ("5b", "5c", "5d", "6a", "6b", "6c", "6d", "6e", "7a", "7b", "7c")
    ops.extend(_mixed(getattr(net, f"Mixed_{m}"), dtype) for m in mixed)
    return _chain(*ops), _linear(net.fc, dtype)
