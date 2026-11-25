"""RTMPose / RTMW on Apple MLX, written in `mlx.core` operations (`MlxRTMPose(net, dtype)`): crops -> SimCC logits.

The maths of `net.RTMPose.forward`, channels last, on the network with its batch norms folded. Written out instead of
replayed from an exported graph: the export and its translation were most of the first call (1.1-1.9 s).

- CSPNeXt (RTMW: and its PAFPN neck) in the run dtype: 1 x 1 convs are matrix products, the 5 x 5 depthwise convs the
  shared depthwise kernel (`graph_mlx.ops.depthwise`; `mx.conv2d(groups=C)` falls off its fast path), the channel
  attention's spatial mean float32; the SPP's 9 x 9 and 13 x 13 max pools are the 5 x 5 one applied two and three
  times (same result: a max of maxes over the same window, the padding -inf).
- SiLU computed in float32: rtmw-l-384 float16 keypoints p99 1.30 px (tol 1.2) with a float16 SiLU, 0.79 px with it
  (the exported port's 0.79).
- The SimCC head in float32: its 1D-bin argmax flips half-pixel bins on float16 logits (face6-s keypoints mean
  0.065 px vs tol 0.061 with a float16 head).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops

from .net import NORM_EPS, SHUFFLE

if TYPE_CHECKING:
    from collections.abc import Callable

    _Layer = Callable[[mx.array], mx.array]

__all__ = ["MlxRTMPose"]

POOL = 5  # the SPP's smallest max pool, which the larger ones repeat (`cspnext.SPP_POOLS` = 5, 9, 13)


def _arr(t: Any, dtype: mx.Dtype) -> mx.array:
    a = mx.array(t.detach().double().numpy().astype(np.float32)).astype(dtype)
    mx.eval(a)
    return a


def _silu(x: mx.array) -> mx.array:
    """SiLU computed in float32 (fused into the conv's epilogue by `mx.compile`)."""
    y = x.astype(mx.float32)
    return (y * mx.sigmoid(y)).astype(x.dtype)


def _conv(conv: Any, dtype: mx.Dtype) -> _Layer:
    """A `Conv2d` (bias folded in) on NHWC maps: 1 x 1 as a matrix product, depthwise as the shared kernel."""
    stride, padding = (int(conv.stride[0]), int(conv.stride[1])), (int(conv.padding[0]), int(conv.padding[1]))
    w, b = conv.weight, _arr(conv.bias, dtype)
    if conv.kernel_size == (1, 1) and stride == (1, 1) and conv.groups == 1:
        m = _arr(w[:, :, 0, 0].t(), dtype)
        return lambda x: x @ m + b
    if conv.groups > 1 and conv.groups == conv.in_channels == conv.out_channels and conv.in_channels % 4 == 0:
        k = _arr(w[:, 0].permute(1, 2, 0), dtype)  # [kh, kw, C]
        return lambda x: ops.depthwise(x, k, b, stride, padding, (1, 1))
    ohwi, groups = _arr(w.permute(0, 2, 3, 1), dtype), conv.groups
    return lambda x: mx.conv2d(x, ohwi, stride=stride, padding=padding, groups=groups) + b


def _conv_act(layer: Any, dtype: mx.Dtype) -> _Layer:
    """`ConvAct`: SiLU(conv(x))."""
    conv = _conv(layer.conv, dtype)
    return lambda x: _silu(conv(x))


def _chain(*ops_: _Layer) -> _Layer:
    def apply(x: mx.array) -> mx.array:
        for op in ops_:
            x = op(x)
        return x

    return apply


def _module(layer: Any, dtype: mx.Dtype) -> _Layer:
    """`ConvAct` or `Separable` (depthwise + SiLU, pointwise + SiLU)."""
    if hasattr(layer, "depthwise_conv"):
        return _chain(_conv_act(layer.depthwise_conv, dtype), _conv_act(layer.pointwise_conv, dtype))
    return _conv_act(layer, dtype)


def _block(block: Any, dtype: mx.Dtype) -> _Layer:
    path = _chain(_module(block.conv1, dtype), _module(block.conv2, dtype))
    return (lambda x: x + path(x)) if block.identity else path


def _attention(att: Any) -> _Layer:
    """Channel attention: x * hardsigmoid(fc(spatial mean)), the mean and the 1 x 1 conv float32."""
    w, b = _arr(att.fc.weight[:, :, 0, 0].t(), mx.float32), _arr(att.fc.bias, mx.float32)

    def apply(x: mx.array) -> mx.array:
        g = mx.mean(x.astype(mx.float32), axis=(1, 2), keepdims=True) @ w + b
        return x * mx.clip(g / 6 + 0.5, 0, 1).astype(x.dtype)

    return apply


def _csp(layer: Any, dtype: mx.Dtype) -> _Layer:
    main, short = _conv_act(layer.main_conv, dtype), _conv_act(layer.short_conv, dtype)
    blocks = _chain(*(_block(b, dtype) for b in layer.blocks))
    attention = _attention(layer.attention) if layer.attention is not None else None
    final = _conv_act(layer.final_conv, dtype)

    def apply(x: mx.array) -> mx.array:
        y = mx.concatenate([blocks(main(x)), short(x)], axis=-1)
        return final(attention(y) if attention is not None else y)

    return apply


def _max_pool(x: mx.array) -> mx.array:
    """`F.max_pool2d(x, 5, 1, 2)` on NHWC: separable maxima of shifted slices of the -inf-padded map."""
    r = POOL // 2
    h, w = x.shape[1], x.shape[2]
    y = mx.pad(x, [(0, 0), (r, r), (0, 0), (0, 0)], constant_values=-math.inf)
    x = y[:, :h]
    for i in range(1, POOL):
        x = mx.maximum(x, y[:, i : i + h])
    y = mx.pad(x, [(0, 0), (0, 0), (r, r), (0, 0)], constant_values=-math.inf)
    x = y[:, :, :w]
    for i in range(1, POOL):
        x = mx.maximum(x, y[:, :, i : i + w])
    return x


def _spp(spp: Any, dtype: mx.Dtype) -> _Layer:
    first, second = _conv_act(spp.conv1, dtype), _conv_act(spp.conv2, dtype)

    def apply(x: mx.array) -> mx.array:
        maps = [first(x)]
        for _ in range(3):  # 5, 9, 13
            maps.append(_max_pool(maps[-1]))
        return second(mx.concatenate(maps, axis=-1))

    return apply


def _stage(stage: Any, dtype: mx.Dtype) -> _Layer:
    """The stride-2 conv, the SPP (last stage), the `CSPLayer`."""
    build = {"CSPLayer": _csp, "SPP": _spp}
    return _chain(*(build.get(type(layer).__name__, _module)(layer, dtype) for layer in stage))


def _up2(x: mx.array) -> mx.array:
    """Nearest x2 upsampling of an NHWC map."""
    n, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None], (n, h, 2, w, 2, c)).reshape(n, 2 * h, 2 * w, c)


def _neck(neck: Any, dtype: mx.Dtype) -> Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array]]:
    """`PAFPN`: (stride 8, 16, 32 maps) -> (stride-16 map, stride-32 map)."""
    reduce = [_conv_act(m, dtype) for m in neck.reduce_layers]
    down = [_conv_act(m, dtype) for m in neck.downsamples]
    top = [_csp(m, dtype) for m in neck.top_down_blocks]
    bottom = [_csp(m, dtype) for m in neck.bottom_up_blocks]

    def apply(f3: mx.array, f4: mx.array, f5: mx.array) -> tuple[mx.array, mx.array]:
        t5 = reduce[0](f5)
        t4 = reduce[1](top[0](mx.concatenate([_up2(t5), f4], axis=-1)))
        m3 = top[1](mx.concatenate([_up2(t4), f3], axis=-1))
        o4 = bottom[0](mx.concatenate([down[0](m3), t4], axis=-1))
        return o4, bottom[1](mx.concatenate([down[1](o4), t5], axis=-1))

    return apply


def _scale_norm(norm: Any) -> _Layer:
    scale, g = norm.scale, float(norm.g.detach())

    def apply(x: mx.array) -> mx.array:
        n = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) * scale
        return x / mx.maximum(n, NORM_EPS) * g

    return apply


def _linear(layer: Any) -> _Layer:
    w = _arr(layer.weight.t(), mx.float32)
    return lambda x: x @ w


def _tokens(mlp: Any) -> _Layer:
    """`flatten(2)` of an NHWC map (`[B, K, h w]`), ScaleNorm, the linear layer."""
    norm, lin = _scale_norm(mlp[0]), _linear(mlp[1])

    def apply(x: mx.array) -> mx.array:
        n, h, w, k = x.shape
        return lin(norm(mx.transpose(x, (0, 3, 1, 2)).reshape(n, k, h * w)))

    return apply


def _gau(gau: Any) -> _Layer:
    e, s = gau.e, gau.gamma.shape[1]
    norm, uv, o = _scale_norm(gau.ln), _linear(gau.uv), _linear(gau.o)
    gamma, beta = _arr(gau.gamma, mx.float32), _arr(gau.beta, mx.float32)
    res, inv = _arr(gau.res_scale.scale, mx.float32), 1 / math.sqrt(s)

    def apply(x: mx.array) -> mx.array:
        y = _silu(uv(norm(x)))
        u, v, base = y[..., :e], y[..., e : 2 * e], y[..., 2 * e :]
        q, k = base * gamma[0] + beta[0], base * gamma[1] + beta[1]
        kernel = mx.square(mx.maximum((q @ mx.swapaxes(k, 1, 2)) * inv, 0))
        return x * res + o(u * (kernel @ v))

    return apply


def _pixel_shuffle(x: mx.array) -> mx.array:
    """`F.pixel_shuffle(x, 2)` on NHWC: channel `c r^2 + i r + j` -> pixel `(r h + i, r w + j)` of channel `c`."""
    n, h, w, c = x.shape
    r = SHUFFLE
    y = x.reshape(n, h, w, c // (r * r), r, r)
    return mx.transpose(y, (0, 1, 4, 2, 5, 3)).reshape(n, h * r, w * r, c // (r * r))


def _head(head: Any) -> Callable[[Any], tuple[mx.array, mx.array]]:
    """`Head` / `WholeHead` in float32: the features -> (x logits, y logits)."""
    f32 = mx.float32
    gau, cls_x, cls_y = _gau(head.gau), _linear(head.cls_x), _linear(head.cls_y)
    if not hasattr(head, "mlp2"):
        final, mlp = _conv(head.final_layer, f32), _tokens(head.mlp)

        def tokens(t: mx.array) -> mx.array:
            return mlp(final(t.astype(f32)))

    else:
        relu = [_conv(head.final_layer.conv, f32), _conv(head.final_layer2.conv, f32), _conv(head.conv_dec.conv, f32)]
        final, final2, dec = (lambda x, c=c: mx.maximum(c(x), 0) for c in relu)
        mlp, mlp2 = _tokens(head.mlp), _tokens(head.mlp2)

        def tokens(feats: tuple[mx.array, mx.array]) -> mx.array:
            b, t = (f.astype(f32) for f in feats)
            fine = mlp2(final2(mx.concatenate([dec(_pixel_shuffle(t)), b], axis=-1)))
            return mx.concatenate([mlp(final(t)), fine], axis=-1)

    def apply(feats: Any) -> tuple[mx.array, mx.array]:
        t = gau(tokens(feats))
        return cls_x(t), cls_y(t)

    return apply


class MlxRTMPose:
    """Normalised NHWC crops (float32) -> float32 SimCC logits (x `[B, K, 2 w]`, y `[B, K, 2 h]`)."""

    def __init__(self, net: Any, dtype: str) -> None:
        """The MLX layers of `net` (an `RTMPose` with its batch norms folded); the trunk in `dtype`."""
        d = mx.float16 if dtype == "float16" else mx.float32
        bb = net.backbone
        stem = _chain(*(_module(m, d) for m in bb.stem))
        stages = [_stage(getattr(bb, f"stage{i}"), d) for i in range(1, 5)]
        neck = _neck(net.neck, d) if net.neck is not None else None
        head = _head(net.head)

        def forward(x: mx.array) -> tuple[mx.array, mx.array]:
            x = stages[0](stem(x.astype(d)))
            c3 = stages[1](x)
            c4 = stages[2](c3)
            c5 = stages[3](c4)
            return head(neck(c3, c4, c5) if neck is not None else c5)

        self.forward = mx.compile(forward)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """SimCC logits of normalised crops `[B, h, w, 3]`."""
        return self.forward(x)
