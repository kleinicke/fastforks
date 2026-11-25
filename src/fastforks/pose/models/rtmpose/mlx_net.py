"""RTMPose / RTMW on Apple MLX, written in `mlx.core` operations (`MlxRTMPose(net, dtype)`): crops -> SimCC logits.

The maths of `net.RTMPose.forward`, channels last, on the network with its batch norms folded. Written out instead of
replayed from an exported graph: the export and its translation were most of the first call (1.1-1.9 s).

- CSPNeXt (RTMW: and its PAFPN neck) in the run dtype: `fastforks.shared.cspnext.mlx_net` (SiLU in float32).
- The SimCC head in float32: its 1D-bin argmax flips half-pixel bins on float16 logits (face6-s keypoints mean
  0.065 px vs tol 0.061 with a float16 head).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.cspnext import mlx_net as csp
from fastforks.shared.cspnext.mlx_net import arr as _arr
from fastforks.shared.cspnext.mlx_net import conv as _conv
from fastforks.shared.cspnext.mlx_net import silu as _silu

from .net import NORM_EPS, SHUFFLE

if TYPE_CHECKING:
    from collections.abc import Callable

    _Layer = Callable[[mx.array], mx.array]

__all__ = ["MlxRTMPose"]


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
        backbone = csp.backbone(net.backbone, d)
        neck = csp.neck(net.neck, d) if net.neck is not None else None
        head = _head(net.head)

        def forward(x: mx.array) -> tuple[mx.array, mx.array]:
            c3, c4, c5 = backbone(x.astype(d))
            return head(neck(c3, c4, c5)[1:] if neck is not None else c5)

        self.forward = mx.compile(forward)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """SimCC logits of normalised crops `[B, h, w, 3]`."""
        return self.forward(x)
