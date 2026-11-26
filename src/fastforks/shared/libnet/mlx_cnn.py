"""Image classifiers in MLX. Convert the eval network once, folding batch norms into OHWI convolutions."""

# ruff: noqa: ANN401 - timm and torchvision register architecture-specific layers

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.shared import mlx_port
from fastforks.shared.libnet import build, skeleton
from fastforks.shared.mlx_graph import conv1x1, conv1x1_ok
from fastforks.shared.mlx_wrap import lazy_build

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.runner import Context


def _mx(t: torch.Tensor, dtype: mx.Dtype) -> mx.array:
    """`t` as an evaluated `mx.array` in `dtype`, or its cached array (`mlx_port.tensor`: the recorded families)."""

    def convert() -> mx.array:
        a = to_mx(t).astype(dtype)
        mx.eval(a)
        return a

    return mlx_port.tensor(t, dtype, convert)


def _conv(conv: nn.Conv2d, bn: nn.BatchNorm2d | None, dtype: mx.Dtype, *, dw_kernel: bool = False) -> Callable:
    """Conv (+ eval batch norm) on NHWC.

    `dw_kernel`: a depthwise one as `graph_mlx`'s kernel (float32 accumulation, and never slower than
    `mx.conv2d(groups=C)`, which falls off its fast path at some channel counts).
    """
    folded = torch.nn.utils.fuse_conv_bn_eval(conv, bn) if bn is not None else conv
    bias = _mx(folded.bias, dtype) if folded.bias is not None else mx.zeros((conv.out_channels,), dtype)
    stride = (conv.stride[0], conv.stride[1])
    padding = (int(conv.padding[0]), int(conv.padding[1]))
    groups = conv.groups
    if conv1x1_ok(conv.in_channels, conv.out_channels, 0, conv.kernel_size, stride, padding, groups):
        # the pointwise conv as one GEMM, its weights stored `[in, out]` (a transposed view is copied on every call)
        matrix = _mx(folded.weight[:, :, 0, 0].t().contiguous(), dtype)
        mx.eval(matrix, bias)
        return lambda x: conv1x1(x, matrix, bias)
    if (
        dw_kernel
        and groups == conv.in_channels == conv.out_channels
        and groups % 4 == 0
        and conv.kernel_size[0] * conv.kernel_size[1] > 1
        and conv.dilation == (1, 1)
    ):
        from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415 - Metal only

        taps = _mx(folded.weight[:, 0].permute(1, 2, 0), dtype)
        mx.eval(taps, bias)
        return lambda x: ops.depthwise(x, taps, bias, stride, padding, (1, 1))
    weight = _mx(folded.weight.permute(0, 2, 3, 1), dtype)
    mx.eval(weight, bias)

    def apply(x: mx.array) -> mx.array:
        return mx.conv2d(x, weight, stride=stride, padding=padding, groups=groups) + bias

    return apply


def _conv_k(conv: nn.Conv2d, bn: nn.BatchNorm2d | None, dtype: mx.Dtype) -> Callable:
    """`_conv` with the depthwise kernel (the newer ports' default)."""
    return _conv(conv, bn, dtype, dw_kernel=True)


def _residual(block: Any, dtype: mx.Dtype) -> Callable:
    first = _conv(block.conv1, block.bn1, dtype)
    second = _conv(block.conv2, block.bn2, dtype)
    third = _conv(block.conv3, block.bn3, dtype) if hasattr(block, "conv3") else None
    down = _conv(*block.downsample, dtype) if block.downsample is not None else None

    def apply(x: mx.array) -> mx.array:
        y = second(mx.maximum(first(x), 0))
        if third is not None:
            y = third(mx.maximum(y, 0))
        return mx.maximum(y + (x if down is None else down(x)), 0)

    return apply


def _resnet(net: Any, dtype: mx.Dtype) -> Callable:
    stem = _conv(net.conv1, net.bn1, dtype)
    blocks = [_residual(b, dtype) for stage in (net.layer1, net.layer2, net.layer3, net.layer4) for b in stage]

    def apply(x: mx.array) -> mx.array:
        x = mx.maximum(stem(x), 0)
        # Zero padding is exact after ReLU. The torchvision max pool is 3x3, stride 2.
        n, h, w, c = x.shape
        x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        windows = mx.as_strided(
            x,
            (n, (h + 1) // 2, (w + 1) // 2, 3, 3, c),
            ((h + 2) * (w + 2) * c, 2 * (w + 2) * c, 2 * c, (w + 2) * c, c, 1),
        )
        x = mx.max(windows, axis=(3, 4))
        for block in blocks:
            x = block(x)
        return x

    return apply


def _mobile_block(block: Any, dtype: mx.Dtype) -> Callable:
    # Each inverted residual ends in a linear pointwise conv/BN. Earlier conv/BN pairs use ReLU6.
    activated = [_conv(layer[0], layer[1], dtype) for layer in list(block.conv)[:-2]]
    project = _conv(block.conv[-2], block.conv[-1], dtype)
    residual = block.use_res_connect

    def apply(x: mx.array) -> mx.array:
        y = x
        for conv in activated:
            y = mx.clip(conv(y), 0, 6)
        y = project(y)
        return x + y if residual else y

    return apply


def _mobilenet(net: Any, dtype: mx.Dtype) -> Callable:
    layers = list(net.features)
    stem = _conv(layers[0][0], layers[0][1], dtype)
    head = _conv(layers[-1][0], layers[-1][1], dtype)
    blocks = [_mobile_block(layer, dtype) for layer in layers[1:-1]]

    def apply(x: mx.array) -> mx.array:
        x = mx.clip(stem(x), 0, 6)
        for block in blocks:
            x = block(x)
        return mx.clip(head(x), 0, 6)

    return apply


def _linear(layer: Any, dtype: mx.Dtype, gamma: torch.Tensor | None = None) -> Callable:
    # Pointwise convolutions and linear layers use the same NHWC matrix multiplication.
    weight, bias = layer.weight.detach().flatten(1), layer.bias.detach()
    if gamma is not None:
        weight, bias = weight * gamma[:, None], bias * gamma
    w, b = _mx(weight.T, dtype), _mx(bias, dtype)
    return lambda x: mx.addmm(b, x, w)


def _norm(layer: Any, dtype: mx.Dtype, *, rows: bool = False) -> Callable:
    """Layer norm over the last axis.

    `rows`: rows up to `ROW_LN_MAX` wide through `graph_mlx`'s row kernel (one simdgroup per row), as the graph
    translation. Faster on ConvNeXt's many-row maps, slower on EfficientFormer's 49-token stage (L1 float16 3.5 ->
    4.0 ms), so opt-in.
    """
    w, b = to_mx(layer.weight).astype(dtype), to_mx(layer.bias).astype(dtype)
    eps = layer.eps
    from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415 - Metal only

    if rows and w.shape[-1] <= ops.ROW_LN_MAX:
        return lambda x: ops.row_layer_norm(x, w, b, eps) if x.dtype == dtype else mx.fast.layer_norm(x, w, b, eps)
    return lambda x: mx.fast.layer_norm(x, w, b, eps)


def _chain(*ops: Callable) -> Callable:
    def apply(x: mx.array) -> mx.array:
        for op in ops:
            x = op(x)
        return x

    apply.ops = ops  # ty: ignore[unresolved-attribute] - read by `_segmented`
    return apply


SEGMENTS = 16
"""Compiled segments of `_segmented`: the top-level ops (stages, blocks) cut into this many of equal output bytes.

M1 Pro, network peak MB / warm ms at 1 / 8 / 16 segments / one per op, graph translation last:
convnext_small 384 float32 443 / 314 / 260 / 256 (26.2 / 26.3 / 26.2 / 27.4 ms), graph 308 (24.5 ms);
davit_small float32 403 / 268 / 244 / 245 (17.0 / 16.6 / 16.5 / 17.9 ms), graph 279 (15.7 ms);
efficientformerv2_l float16 107 / 79 / 70 / 57 (8.7 / 8.4 / 8.4 / 9.7 ms), graph 101 (9.2 ms).
"""
SEGMENT_MIN_OPS = 3
"""Top-level ops per segment at least: each wait costs a GPU round trip, which a small network feels
(efficientformerv2_s0 float16: 3.3 ms at 16 segments, 3.0 at 8, the graph translation 3.0)."""


SYNC_TOTAL = 32 << 20
"""Ops' output bytes per call from which a `sync` network's segments each wait for themselves: two segments in
flight held both's transients, which in TinyViT's attention stages are several times the segment's outputs
(tiny_vit_21m_512 float32, 63 MB of outputs: network peak 1.30 -> 1.00x the graph translation's). Only TinyViT: on
ConvNeXt the overlap is worth more than the transients (convnext_base 384 float32 warm 0.73 -> 1.02x the graph's for
the same peak, convnext_large 224 float32 0.89 -> 1.03x)."""

SYNC_FAMILIES = frozenset({"tinyvit"})
"""The families whose `_segmented` segments run synchronously from `SYNC_TOTAL` on."""


def _segmented(features: Callable, tail: Callable, *, sync: bool = False) -> Callable:
    """`tail(features(x))` as compiled segments of `features`' ops, evaluated one after the other (one in flight).

    MLX returns a command buffer's transients only when it completes: one compiled function over a whole network
    holds every block's intermediates until the end (up to twice the graph translation's peak, which evaluates every
    256 MB of activations). Per input shape the ops' outputs are sized lazily (nothing runs) and cut into `SEGMENTS`
    of about equal output bytes; a block's transients are a multiple of its output, so each segment holds a share.
    """
    found = getattr(features, "ops", None)
    if found is None:
        return mx.compile(lambda x: tail(features(x)))
    ops: tuple[Callable, ...] = tuple(found)
    plans: dict[tuple[tuple[int, ...], mx.Dtype], tuple[list[Callable], bool]] = {}

    def plan(x: mx.array) -> tuple[list[Callable], bool]:
        sizes = []
        for op in ops:
            x = op(x)
            sizes.append(x.nbytes)
        del x
        cuts, pending, share = [0], 0, sum(sizes) / SEGMENTS
        for i, n in enumerate(sizes[:-1]):
            pending += n
            if pending >= share and i + 1 - cuts[-1] >= SEGMENT_MIN_OPS:
                cuts.append(i + 1)
                pending = 0
        cuts.append(len(ops))
        segs = [_chain(*ops[a:b]) for a, b in itertools.pairwise(cuts)]
        last = segs[-1]
        segs[-1] = lambda x: tail(last(x))
        return [mx.compile(s) for s in segs], sync and sum(sizes) >= SYNC_TOTAL

    def run(x: mx.array) -> mx.array:
        key = (tuple(x.shape), x.dtype)
        found = plans.get(key)
        if found is None:
            found = plans[key] = plan(x)
        segs, sync = found
        prev = None
        for seg in segs[:-1]:
            x = seg(x)
            (mx.eval if sync else mx.async_eval)(x)
            if prev is not None:
                mx.eval(prev)
            prev = x
        return segs[-1](x)

    return run


def _grn(layer: Any, dtype: mx.Dtype) -> Callable:
    weight, bias, eps = to_mx(layer.weight), to_mx(layer.bias), layer.eps

    def apply(x: mx.array) -> mx.array:
        value = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
        scale = norm / (mx.mean(norm, axis=-1, keepdims=True) + eps)
        return (value + bias + weight * value * scale).astype(dtype)

    return apply


def _gelu(x: mx.array) -> mx.array:
    return x * (0.5 + 0.5 * mx.erf(x * 0.7071067811865476))


def _gelu_f32(x: mx.array) -> mx.array:
    """GELU computed in float32 (`0.5 + 0.5 * erf` of a negative float16 input cancels), rounded once."""
    y = x.astype(mx.float32)
    return (y * (0.5 + 0.5 * mx.erf(y * 0.7071067811865476))).astype(x.dtype)


_gelu_f32 = mx.compile(_gelu_f32, shapeless=True)


def _convnext_block(block: Any, dtype: mx.Dtype) -> Callable:
    act = block.mlp.act
    tanh = type(act).__name__ == "GELUTanh" or getattr(act, "approximate", "none") == "tanh"  # the WD tagger's
    from .mlx_features import gelu_tanh  # noqa: PLC0415 - the WD tagger's (mlx_features imports timm's helpers)

    ops = [_conv_k(block.conv_dw, None, dtype), _norm(block.norm, dtype, rows=True), _linear(block.mlp.fc1, dtype)]
    ops.append(gelu_tanh if tanh else _gelu_f32)
    if hasattr(block.mlp, "grn"):
        ops.append(_grn(block.mlp.grn, dtype))
    path = _chain(*ops, _linear(block.mlp.fc2, dtype, block.gamma))
    return _residual_path(path)


def _convnext_trunk(net: Any, dtype: mx.Dtype) -> Callable:
    ops = [_conv(net.stem[0], None, dtype), _norm(net.stem[1], dtype)]
    for stage in net.stages:
        if not isinstance(stage.downsample, nn.Identity):
            ops.extend((_norm(stage.downsample[0], dtype), _conv(stage.downsample[1], None, dtype)))
        ops.extend(_convnext_block(block, dtype) for block in stage.blocks)
    return _chain(*ops)


def _activation(layer: nn.Module) -> Callable:
    if isinstance(layer, nn.ReLU6):
        return lambda x: mx.clip(x, 0, 6)
    if isinstance(layer, nn.ReLU):
        return lambda x: mx.maximum(x, 0)
    if isinstance(layer, nn.SiLU):
        return lambda x: x * mx.sigmoid(x)
    if isinstance(layer, nn.Hardswish):
        return lambda x: x * (mx.clip(x + 3, 0, 6) / 6)
    if isinstance(layer, nn.Hardsigmoid):
        return lambda x: mx.clip(x + 3, 0, 6) / 6
    if isinstance(layer, nn.Sigmoid):
        return mx.sigmoid
    msg = f"Unsupported MLX activation {type(layer).__name__}"
    raise ValueError(msg)


def _conv_norm(layer: Any, dtype: mx.Dtype) -> Callable:
    conv = _conv(layer[0], layer[1], dtype)
    return _chain(conv, *(_activation(act) for act in list(layer)[2:]))


def _excitation(layer: Any, dtype: mx.Dtype) -> Callable:
    gate = _chain(
        _linear(layer.fc1, dtype),
        _activation(layer.activation),
        _linear(layer.fc2, dtype),
        _activation(layer.scale_activation),
    )
    return lambda x: x * gate(_mean_spatial(x))[:, None, None, :]


def _bottleneck(layers: Any, dtype: mx.Dtype) -> Callable:
    return _chain(
        *(
            _excitation(layer, dtype) if hasattr(layer, "scale_activation") else _conv_norm(layer, dtype)
            for layer in layers
        )
    )


def _mobile_residual(block: Any, dtype: mx.Dtype) -> Callable:
    path = _bottleneck(block.block, dtype)
    return (lambda x: x + path(x)) if block.use_res_connect else path


def _regnet_block(block: Any, dtype: mx.Dtype) -> Callable:
    path = _bottleneck(block.f, dtype)
    projection = _conv_norm(block.proj, dtype) if block.proj is not None else None
    return lambda x: mx.maximum(path(x) + (projection(x) if projection else x), 0)


def _gated_cnn(net: Any, dtype: mx.Dtype, *, regnet: bool) -> tuple[Callable, Callable]:
    if regnet:
        blocks = [_regnet_block(b, dtype) for stage in net.trunk_output for b in stage]
        return _chain(_conv_norm(net.stem, dtype), *blocks), _linear(net.fc, dtype)
    blocks = []
    for stage in list(net.features)[1:-1]:
        # MobileNetV3 exposes individual blocks, EfficientNet groups them by stage.
        blocks.extend(_mobile_residual(b, dtype) for b in ([stage] if hasattr(stage, "block") else stage))
    features = _chain(_conv_norm(net.features[0], dtype), *blocks, _conv_norm(net.features[-1], dtype))
    head = [
        _linear(layer, dtype) if isinstance(layer, nn.Linear) else _activation(layer)
        for layer in net.classifier
        if not isinstance(layer, nn.Dropout)
    ]
    return features, _chain(*head)


def _mean_spatial(x: mx.array) -> mx.array:
    # The excitation gates reduce large feature maps; parallelise over pixels with MLX's native reduction.
    return mx.mean(x.astype(mx.float32), axis=(1, 2)).astype(x.dtype)


def _pool_for(mode: str) -> Callable:
    if mode == "kernels":
        from fastforks.kernels.cnn_mlx.ops import pool  # noqa: PLC0415 - only in kernels mode

        return pool
    return _mean_spatial


class ConvNextFeatures:
    """A ConvNeXt trunk returning row-major patch tokens and the normalised pooled embedding."""

    def __init__(self, net: Any, ctx: Context) -> None:
        """Convert the backbone weights after the caller has removed the classifier."""
        self.dtype = getattr(mx, ctx.dtype)
        trunk = _convnext_trunk(net, self.dtype)
        norm, pool = _norm(net.head.norm, mx.float32), _pool_for(ctx.mode)

        def forward(x: mx.array) -> tuple[mx.array, mx.array]:
            grid = trunk(x)
            tokens = grid.reshape(grid.shape[0], -1, grid.shape[-1])
            return tokens, norm(pool(grid.astype(mx.float32)))

        self.forward = mx.compile(forward)

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """NCHW normalised images to float32 tokens and pooled embeddings."""
        tokens, pooled = self.forward(to_mx(image.permute(0, 2, 3, 1)).astype(self.dtype))
        return to_torch(tokens), to_torch(pooled)


class DinoConvNext:
    """HF DINOv3 ConvNeXt returns the normalised pooled token followed by normalised grid tokens."""

    def __init__(self, net: Any, ctx: Context) -> None:
        """Convert HF-named weights, folding LayerScale into the second projection."""
        self.dtype = getattr(mx, ctx.dtype)
        ops = []
        for stage in net.stages:
            ops.extend(
                _conv(layer, None, self.dtype) if isinstance(layer, nn.Conv2d) else _norm(layer, self.dtype)
                for layer in stage.downsample_layers
            )
            for block in stage.layers:
                path = _chain(
                    _conv(block.depthwise_conv, None, self.dtype),
                    _norm(block.layer_norm, self.dtype),
                    _linear(block.pointwise_conv1, self.dtype),
                    _gelu,
                    _linear(block.pointwise_conv2, self.dtype, block.gamma),
                )
                ops.append(_residual_path(path))
        trunk, norm, pool = _chain(*ops), _norm(net.layer_norm, mx.float32), _pool_for(ctx.mode)

        def forward(x: mx.array) -> mx.array:
            grid = trunk(x).astype(mx.float32)
            tokens = grid.reshape(grid.shape[0], -1, grid.shape[-1])
            return norm(mx.concatenate([pool(grid)[:, None], tokens], axis=1))

        self.forward = mx.compile(forward)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """NCHW normalised images to float32 pooled and spatial tokens."""
        return to_torch(self.forward(to_mx(image.permute(0, 2, 3, 1)).astype(self.dtype)))


def _residual_path(path: Callable) -> Callable:
    return lambda x: x + path(x)


def _pool2d(layer: Any) -> Callable:
    """Channels-last square pooling, including torchvision's ceil-mode final window."""
    k, step, pad = layer.kernel_size, layer.stride, layer.padding
    maximum = isinstance(layer, nn.MaxPool2d)

    def apply(x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        extra = step - 1 if layer.ceil_mode else 0
        oh, ow = (h + 2 * pad - k + extra) // step + 1, (w + 2 * pad - k + extra) // step + 1
        ph, pw = max(pad, (oh - 1) * step + k - h - pad), max(pad, (ow - 1) * step + k - w - pad)
        padded = mx.contiguous(
            mx.pad(x, [(0, 0), (pad, ph), (pad, pw), (0, 0)], constant_values=-float("inf") if maximum else 0)
        )
        row = padded.shape[2] * c
        windows = mx.as_strided(padded, (n, oh, ow, k, k, c), (padded.shape[1] * row, step * row, step * c, row, c, 1))
        return mx.max(windows, axis=(3, 4)) if maximum else mx.mean(windows, axis=(3, 4))

    return apply


def _classic_op(layer: Any, dtype: mx.Dtype) -> Callable:
    """Convert a torchvision classifier's eval layers to channels-last operations."""
    if isinstance(layer, nn.Conv2d):
        return _conv_k(layer, None, dtype)
    if isinstance(layer, nn.BatchNorm2d):
        scale = layer.weight.detach().float() / (layer.running_var.float() + layer.eps).sqrt()
        gain, bias = to_mx(scale).astype(dtype), to_mx(layer.bias - layer.running_mean * scale).astype(dtype)
        return lambda x: x * gain + bias
    if isinstance(layer, nn.Sequential):
        return _chain(*(_classic_op(m, dtype) for m in layer))
    if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
        return _pool2d(layer)
    if isinstance(layer, nn.Linear):
        return _linear(layer, dtype)
    return _classic_block(layer, dtype)


def _classic_block(layer: Any, dtype: mx.Dtype) -> Callable:
    if isinstance(layer, nn.Dropout):
        return lambda x: x
    if hasattr(layer, "expand1x1"):  # SqueezeNet Fire
        squeeze, left, right = (_classic_op(m, dtype) for m in (layer.squeeze, layer.expand1x1, layer.expand3x3))

        def fire(x: mx.array) -> mx.array:
            x = mx.maximum(squeeze(x), 0)
            return mx.maximum(mx.concatenate([left(x), right(x)], axis=-1), 0)

        return fire
    if hasattr(layer, "denselayer1"):  # DenseNet concatenates every layer's output to all preceding features
        dense = [_chain(*(_classic_op(m, dtype) for m in unit.children())) for unit in layer.children()]

        def dense_block(x: mx.array) -> mx.array:
            for fn in dense:
                x = mx.concatenate([x, fn(x)], axis=-1)
            return x

        return dense_block
    if hasattr(layer, "branch2"):  # ShuffleNetV2 inverted residual
        first, second = _classic_op(layer.branch1, dtype), _classic_op(layer.branch2, dtype)
        split = layer.stride == 1

        def shuffle(x: mx.array) -> mx.array:
            a, b = mx.split(x, 2, axis=-1) if split else (first(x), x)
            y = mx.concatenate([a, second(b)], axis=-1)
            return y.reshape(*y.shape[:-1], 2, -1).swapaxes(-2, -1).reshape(y.shape)

        return shuffle
    return _activation(layer)


def _classic(net: Any, family: str, dtype: mx.Dtype) -> Callable:
    if family == "shufflenetv2":
        trunk = _classic_op(nn.Sequential(net.conv1, net.maxpool, net.stage2, net.stage3, net.stage4, net.conv5), dtype)
        return _chain(trunk, _mean_spatial, _linear(net.fc, dtype))
    trunk = _classic_op(net.features, dtype)
    if family == "densenet":
        return _chain(trunk, lambda x: mx.maximum(x, 0), _mean_spatial, _linear(net.classifier, dtype))
    if family == "squeezenet":
        return _chain(trunk, _classic_op(net.classifier[:-1], dtype), _mean_spatial)
    # AlexNet and VGG use fixed 224px crops; the adaptive pool has the same input and output shape.
    return _chain(trunk, lambda x: x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1), _classic_op(net.classifier, dtype))


def _mobile_v1(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    blocks = [_conv_norm_act(net.conv_stem, net.bn1, dtype)]
    for stage in net.blocks:
        for block in stage:
            blocks.extend(
                (_conv_norm_act(block.conv_dw, block.bn1, dtype), _conv_norm_act(block.conv_pw, block.bn2, dtype))
            )
    return _chain(*blocks), _linear(net.classifier, dtype)


def _conv_norm_act(conv: nn.Conv2d, bn: nn.BatchNorm2d, dtype: mx.Dtype) -> Callable:
    # `_conv`'s depthwise path: with `_conv_k` MobileNetV1 ran 1.15-1.22x the graph translation (0.95-1.02 before)
    return _chain(_conv(conv, bn, dtype), _activation(cast("nn.Module", bn.act)))


def _star_conv(layer: Any, dtype: mx.Dtype) -> Callable:
    return _conv(layer.conv, getattr(layer, "bn", None), dtype)


def _star_block(block: Any, dtype: mx.Dtype) -> Callable:
    depth, left, right, project, finish = (
        _star_conv(layer, dtype) for layer in (block.dwconv, block.f1, block.f2, block.g, block.dwconv2)
    )

    def run(x: mx.array) -> mx.array:
        y = depth(x)
        return x + finish(project(mx.clip(left(y), 0, 6) * right(y)))

    return run


def _star_net(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    ops = [_star_conv(net.stem[0], dtype), _activation(net.stem[1])]
    for stage in net.stages:
        ops.append(_star_conv(stage[0], dtype))
        ops.extend(_star_block(block, dtype) for block in list(stage)[1:])
    ops.append(_classic_op(net.norm, dtype))
    return _chain(*ops), _linear(net.head, dtype)


def _rep_mlp(layer: Any, dtype: mx.Dtype) -> Callable:
    return _residual_path(_chain(_conv(layer.conv1, None, dtype), _gelu, _conv(layer.conv2, None, dtype)))


def _rep_block(block: Any, dtype: mx.Dtype) -> Callable:
    ops = [_conv(block.token_mixer, None, dtype)]
    if not isinstance(block.se, nn.Identity):
        gate = _chain(_linear(block.se.fc1, dtype), _activation(block.se.act), _linear(block.se.fc2, dtype), mx.sigmoid)
        ops.append(lambda x: x * gate(_mean_spatial(x))[:, None, None, :])
    return _chain(*ops, _rep_mlp(block.channel_mixer, dtype))


def _rep_vit(net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:
    # Timm folds the parallel depthwise branches, BatchNorms and the two distillation heads.
    net.fuse()
    ops = [_conv(net.stem.conv1, None, dtype), _gelu, _conv(net.stem.conv2, None, dtype)]
    for stage in net.stages:
        down = stage.downsample
        if not isinstance(down, nn.Identity):
            ops.extend(
                (
                    _rep_block(down.pre_block, dtype),
                    _conv(down.spatial_downsample, None, dtype),
                    _conv(down.channel_downsample, None, dtype),
                    _rep_mlp(down.ffn, dtype),
                )
            )
        ops.extend(_rep_block(block, dtype) for block in stage.blocks)
    return _chain(*ops), _linear(net.head, dtype)


# Families with a hand-written port here; every other network takes the generic translation (`mlx_graph`)
FAMILIES = frozenset(
    {
        "alexnet", "vgg", "densenet", "squeezenet", "shufflenetv2", "convnext", "convnextv2", "resnet", "mobilenetv2",
        "efficientnet", "efficientnetv2", "mobilenetv3", "regnet", "mobilenetv1", "starnet", "repvit",
        "coatnet", "efficientformerv2", "hgnetv2", "focalnet", "swin", "davit", "tinyvit", "mobilenetv4",
        "inception_v3", "efficientvit", "shvit", "efficientformer", "fastvit",
    }
)  # fmt: skip


# `(features, head)` builders of the networks written out here, and of those in their own modules (named by family)
_BUILDERS = {"mobilenetv1": _mobile_v1, "starnet": _star_net, "repvit": _rep_vit}


# Families whose ports live in their own modules (which import this one, so they load lazily)
_MODULES = frozenset(
    {
        "coatnet",
        "efficientformerv2",
        "hgnetv2",
        "focalnet",
        "swin",
        "davit",
        "tinyvit",
        "mobilenetv4",
        "inception_v3",
        "efficientvit",
        "shvit",
        "efficientformer",
        "fastvit",
    }
)


def _module_network(family: str, net: Any, dtype: mx.Dtype) -> tuple[Callable, Callable]:  # noqa: C901, PLR0911
    if family == "coatnet":
        from .mlx_coatnet import coatnet  # noqa: PLC0415

        return coatnet(net, dtype)
    if family == "efficientformerv2":
        from .mlx_efficientformerv2 import efficientformerv2  # noqa: PLC0415

        return efficientformerv2(net, dtype)
    if family == "hgnetv2":
        from .mlx_hgnetv2 import hgnetv2  # noqa: PLC0415

        return hgnetv2(net, dtype)
    if family == "focalnet":
        from .mlx_focalnet import focalnet  # noqa: PLC0415

        return focalnet(net, dtype)
    if family in {"swin", "davit"}:
        from . import mlx_windowed  # noqa: PLC0415

        return getattr(mlx_windowed, family)(net, dtype)
    if family == "tinyvit":
        from .mlx_tinyvit import tinyvit  # noqa: PLC0415

        return tinyvit(net, dtype)
    if family == "mobilenetv4":
        from .mlx_mobilenetv4 import mobilenetv4  # noqa: PLC0415

        return mobilenetv4(net, dtype)
    if family == "inception_v3":
        from .mlx_inception import inception  # noqa: PLC0415

        return inception(net, dtype)
    if family == "efficientvit":
        from .mlx_efficientvit import efficientvit  # noqa: PLC0415

        return efficientvit(net, dtype)
    if family == "shvit":
        from .mlx_shvit import shvit  # noqa: PLC0415

        return shvit(net, dtype)
    if family == "efficientformer":
        from .mlx_efficientformer import efficientformer  # noqa: PLC0415

        return efficientformer(net, dtype)
    from .mlx_fastvit import fastvit  # noqa: PLC0415

    return fastvit(net, dtype)


# Families whose builder reads every weight through `_mx`: their arrays come from the conversion cache
# (`mlx_port.recorded`), so a warm load builds a weightless skeleton and reads no checkpoint
_RECORDED = {"resnet": "cnn-resnet-v1", "mobilenetv2": "cnn-mobilenetv2-v1"}


class Network:
    """Fold and upload the weights once, then run the full classifier in MLX."""

    def __init__(self, ctx: Context, net: torch.nn.Module | None = None, family: str | None = None) -> None:
        """Convert the selected network from timm or torchvision.

        `net`: the family's network (default: `build(ctx)`'s lazy twin); `family`: the architecture to write it out
        as (default: the card's family; the WD tagger's ConvNeXt is `"convnext"`).
        """
        family = family or ctx.card.family
        self.dtype = getattr(mx, ctx.dtype)
        if net is None and family in _RECORDED:
            self.forward = mlx_port.recorded(
                ctx,
                _RECORDED[family],
                lambda net: self._forward(net, family, ctx.mode),
                real=lambda: build(ctx),
                skeleton=lambda: skeleton(ctx),
                sources=(Path(__file__),),
            )
            return
        # timm and torchvision register architecture-specific layers; the weights stay the memory-mapped checkpoint
        net = cast("Any", net if net is not None else lazy_build(lambda: build(ctx)))
        self.forward = self._forward(net, family, ctx.mode)

    def _forward(self, net: Any, family: str, mode: str) -> Callable:
        """The compiled MLX forward of `net` written out as `family`."""
        if family in {"alexnet", "vgg", "densenet", "squeezenet", "shufflenetv2"}:
            return mx.compile(_classic(net, family, self.dtype))
        pool = _pool_for(mode)
        if family in {"convnext", "convnextv2"}:
            features = _convnext_trunk(net, self.dtype)
            head = _chain(_norm(net.head.norm, self.dtype), _linear(net.head.fc, self.dtype))
        elif family == "resnet":
            features, head = _resnet(net, self.dtype), _linear(net.fc, self.dtype)
        elif family == "mobilenetv2":
            features, head = _mobilenet(net, self.dtype), _linear(net.classifier[-1], self.dtype)
        elif family in {"efficientnet", "efficientnetv2", "mobilenetv3", "regnet"}:
            features, head = _gated_cnn(net, self.dtype, regnet=family == "regnet")
        elif family in _BUILDERS:
            features, head = _BUILDERS[family](net, self.dtype)
        elif family in _MODULES:
            features, head = _module_network(family, net, self.dtype)
        else:
            msg = f"No MLX classifier implementation for {family}"
            raise ValueError(msg)

        return _segmented(features, lambda x: head(pool(x)), sync=family in SYNC_FAMILIES)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """NCHW normalised images to float32 logits."""
        return to_torch(self.forward(to_mx(image.permute(0, 2, 3, 1)).astype(self.dtype)))
