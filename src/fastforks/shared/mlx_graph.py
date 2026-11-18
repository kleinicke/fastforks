"""Generic PyTorch -> MLX translation: export an eval network to core ATen and replay it on `mlx.core`.

The hand-written MLX ports (`mlx_cnn`, `mlx_vit`, the stereo / DPT / RoMa ports) give the best speed for a family, but
a port per architecture does not scale to every network. This module covers the rest: `GraphNet(module, dtype)`
exports the (fused, eval-mode) module once per input shape with `torch.export`, decomposes it to core ATen, converts
every parameter once to an `mx.array` of the run dtype, and replays the graph through a table of `mx` ops under
`mx.compile`.

What the translation does beyond a 1:1 op mapping:
- Convolutions run channels-last (MLX's layout): a conv is `transpose -> mx.conv -> transpose`; neighbouring
  transposes cancel in the lazy graph, so a conv trunk is transposed at its ends only.
- `BatchNorm` in eval mode directly after a convolution is folded into the convolution's weight and bias.
- `scaled_dot_product_attention` stays one op (`mx.fast.scaled_dot_product_attention`).
- Float parameters are stored in the run dtype; norms, softmax and reductions accumulate in float32.

An op without a mapping raises `UnsupportedOp` naming it (never a silent fallback).
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.mlx import mx, to_mx, to_torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["GraphNet", "UnsupportedOp"]

aten = torch.ops.aten


class UnsupportedOp(NotImplementedError):  # noqa: N818 - reads better as a noun at the raise site
    """The exported graph contains an ATen op the MLX translation has no mapping for."""


_DTYPES = {
    torch.float32: mx.float32,
    torch.float16: mx.float16,
    torch.bfloat16: mx.bfloat16,
    torch.int64: mx.int64,
    torch.int32: mx.int32,
    torch.int16: mx.int16,
    torch.int8: mx.int8,
    torch.uint8: mx.uint8,
    torch.bool: mx.bool_,
    torch.float64: mx.float32,
}


def _dt(d: torch.dtype | None) -> mx.Dtype | None:
    return None if d is None else _DTYPES[d]


OPS: dict[Any, Callable[..., Any]] = {}


def _reg(*targets: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # noqa: ANN401
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        for t in targets:
            OPS[t] = fn
        return fn

    return deco


def _a(*names: str) -> list[Any]:
    """The overloads of `aten.<name>` that exist."""
    out = []
    for n in names:
        base, _, ov = n.partition(".")
        pkt = getattr(aten, base, None)
        if pkt is None:
            continue
        out.append(getattr(pkt, ov or "default", None) if ov else pkt.default)
    return [o for o in out if o is not None]


# ---- elementwise ------------------------------------------------------------------------------------------------
def _alpha(b: Any, alpha: Any) -> Any:  # noqa: ANN401
    return b if alpha == 1 else b * alpha


OPS[aten.add.Tensor] = lambda a, b, alpha=1: a + _alpha(b, alpha)
OPS[aten.add.Scalar] = OPS[aten.add.Tensor]
OPS[aten.sub.Tensor] = lambda a, b, alpha=1: a - _alpha(b, alpha)
OPS[aten.sub.Scalar] = OPS[aten.sub.Tensor]
OPS[aten.rsub.Scalar] = lambda a, b, alpha=1: b - _alpha(a, alpha)
OPS[aten.rsub.Tensor] = OPS[aten.rsub.Scalar]
OPS[aten.mul.Tensor] = operator.mul
OPS[aten.mul.Scalar] = operator.mul
OPS[aten.div.Tensor] = operator.truediv
OPS[aten.div.Scalar] = operator.truediv
OPS[aten.neg.default] = operator.neg
OPS[aten.abs.default] = mx.abs
OPS[aten.exp.default] = mx.exp
OPS[aten.log.default] = mx.log
OPS[aten.sqrt.default] = mx.sqrt
OPS[aten.rsqrt.default] = mx.rsqrt
OPS[aten.sin.default] = mx.sin
OPS[aten.cos.default] = mx.cos
OPS[aten.tanh.default] = mx.tanh
OPS[aten.sigmoid.default] = mx.sigmoid
OPS[aten.erf.default] = mx.erf
OPS[aten.floor.default] = mx.floor
OPS[aten.ceil.default] = mx.ceil
OPS[aten.round.default] = mx.round
OPS[aten.reciprocal.default] = mx.reciprocal
OPS[aten.sign.default] = mx.sign
OPS[aten.square.default] = mx.square
OPS[aten.relu.default] = lambda x: mx.maximum(x, 0)
OPS[aten.silu.default] = lambda x: x * mx.sigmoid(x)
OPS[aten.hardtanh.default] = lambda x, lo=-1.0, hi=1.0: mx.clip(x, lo, hi)
OPS[aten.hardsigmoid.default] = lambda x: mx.clip(x / 6 + 0.5, 0, 1)
OPS[aten.hardswish.default] = lambda x: x * mx.clip(x / 6 + 0.5, 0, 1)
OPS[aten.leaky_relu.default] = lambda x, s=0.01: mx.where(x >= 0, x, x * s)
OPS[aten.elu.default] = lambda x, a=1.0, *_: mx.where(x > 0, x, a * (mx.exp(x) - 1))
OPS[aten.softplus.default] = lambda x, beta=1.0, thr=20.0: mx.where(x * beta > thr, x, mx.log1p(mx.exp(x * beta)) / beta)
OPS[aten.pow.Tensor_Scalar] = lambda x, e: mx.power(x, e)
OPS[aten.pow.Tensor_Tensor] = mx.power
OPS[aten.pow.Scalar] = lambda b, e: mx.power(b, e)
OPS[aten.maximum.default] = mx.maximum
OPS[aten.minimum.default] = mx.minimum
OPS[aten.clamp.default] = lambda x, lo=None, hi=None: mx.clip(x, lo, hi)
OPS[aten.clamp_min.default] = lambda x, lo: mx.maximum(x, lo)
OPS[aten.clamp_max.default] = lambda x, hi: mx.minimum(x, hi)
OPS[aten.eq.Scalar] = OPS[aten.eq.Tensor] = operator.eq
OPS[aten.ne.Scalar] = OPS[aten.ne.Tensor] = operator.ne
OPS[aten.lt.Scalar] = OPS[aten.lt.Tensor] = operator.lt
OPS[aten.le.Scalar] = OPS[aten.le.Tensor] = operator.le
OPS[aten.gt.Scalar] = OPS[aten.gt.Tensor] = operator.gt
OPS[aten.ge.Scalar] = OPS[aten.ge.Tensor] = operator.ge
OPS[aten.logical_not.default] = mx.logical_not
OPS[aten.logical_and.default] = mx.logical_and
OPS[aten.logical_or.default] = mx.logical_or
OPS[aten.bitwise_not.default] = mx.logical_not
OPS[aten.isnan.default] = mx.isnan
OPS[aten.where.self] = lambda c, a, b: mx.where(c, a, b)
OPS[aten.where.ScalarOther] = OPS[aten.where.ScalarSelf] = OPS[aten.where.self]
OPS[aten.masked_fill.Scalar] = lambda x, m, v: mx.where(m, mx.array(v, x.dtype), x)
OPS[aten.remainder.Scalar] = lambda a, b: mx.remainder(a, b)
def _fmod(a: Any, b: Any) -> Any:  # noqa: ANN401
    r = mx.remainder(a, b)
    return mx.where((r != 0) & ((a < 0) != (b < 0)), r - b, r)


OPS[aten.fmod.Scalar] = OPS[aten.fmod.Tensor] = _fmod
OPS[aten.floor_divide.default] = lambda a, b: mx.floor_divide(a, b)


@_reg(aten.gelu.default)
def _gelu(x: mx.array, approximate: str = "none") -> mx.array:
    if approximate == "tanh":
        c = math.sqrt(2 / math.pi)
        return 0.5 * x * (1 + mx.tanh(c * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))


# ---- shape ------------------------------------------------------------------------------------------------------
OPS[aten.view.default] = OPS[aten._unsafe_view.default] = OPS[aten.reshape.default] = lambda x, s: mx.reshape(x, list(s))
OPS[aten.permute.default] = lambda x, d: mx.transpose(x, list(d))
OPS[aten.transpose.int] = lambda x, a, b: mx.swapaxes(x, a, b)
OPS[aten.t.default] = lambda x: x.T
OPS[aten.unsqueeze.default] = lambda x, d: mx.expand_dims(x, d)
OPS[aten.squeeze.dim] = lambda x, d: mx.squeeze(x, d) if x.shape[d] == 1 else x
OPS[aten.squeeze.dims] = lambda x, d: mx.squeeze(x, [i for i in d if x.shape[i] == 1])
OPS[aten.squeeze.default] = lambda x: mx.squeeze(x)
OPS[aten.flatten.using_ints] = lambda x, s=0, e=-1: mx.flatten(x, s, e)
OPS[aten.clone.default] = OPS[aten.contiguous.default] = OPS[aten.detach.default] = OPS[aten.alias.default] = (
    lambda x, **_: x
)
OPS[aten.lift_fresh_copy.default] = lambda x: x
OPS[aten.cat.default] = lambda xs, d=0: mx.concatenate(list(xs), axis=d)
OPS[aten.stack.default] = lambda xs, d=0: mx.stack(list(xs), axis=d)
OPS[aten.select.int] = lambda x, d, i: mx.take(x, i, axis=d)
OPS[aten.flip.default] = lambda x, dims: _flip(x, dims)
OPS[aten.roll.default] = lambda x, shifts, dims=(): _roll(x, shifts, dims)
OPS[aten.repeat.default] = lambda x, r: mx.tile(x, list(r))
OPS[aten.tril.default] = lambda x, k=0: mx.tril(x, k)
OPS[aten.triu.default] = lambda x, k=0: mx.triu(x, k)
OPS[aten.cumsum.default] = lambda x, d, dtype=None: mx.cumsum(x, d)
OPS[aten.as_strided.default] = lambda x, size, stride, offset=0: mx.as_strided(
    x, list(size), list(stride), int(offset or 0)
)
OPS[aten._assert_tensor_metadata.default] = lambda *a, **k: None
OPS[aten._assert_scalar.default] = lambda *a, **k: None
OPS[aten.sym_size.int] = lambda x, d: x.shape[d]


def _flip(x: mx.array, dims: Sequence[int]) -> mx.array:
    for d in dims:
        x = mx.take(x, mx.arange(x.shape[d] - 1, -1, -1), axis=d)
    return x


def _roll(x: mx.array, shifts: Any, dims: Any) -> mx.array:  # noqa: ANN401
    if not dims:
        return mx.roll(x, list(shifts)[0])
    return mx.roll(x, list(shifts), axis=list(dims))


@_reg(aten.expand.default)
def _expand(x: mx.array, size: Sequence[int], implicit: bool = False) -> mx.array:  # noqa: FBT001, FBT002
    del implicit
    lead = len(size) - x.ndim
    shape = [(x.shape[i - lead] if s == -1 else s) for i, s in enumerate(size)]
    return mx.broadcast_to(x, shape)


@_reg(aten.slice.Tensor)
def _slice(x: mx.array, dim: int = 0, start: int | None = None, end: int | None = None, step: int = 1) -> mx.array:
    n = x.shape[dim]
    start = 0 if start is None else max(0, start + n if start < 0 else start)
    end = n if end is None else min(n, end + n if end < 0 else end)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(start, max(start, end), step)
    return x[tuple(idx)]


@_reg(aten.split.Tensor)
def _split(x: mx.array, size: int, dim: int = 0) -> list[mx.array]:
    return mx.split(x, list(range(size, x.shape[dim], size)), axis=dim)


@_reg(aten.split_with_sizes.default)
def _split_sizes(x: mx.array, sizes: Sequence[int], dim: int = 0) -> list[mx.array]:
    cuts = [sum(sizes[: i + 1]) for i in range(len(sizes) - 1)]
    return mx.split(x, cuts, axis=dim)


@_reg(aten.chunk.default)
def _chunk(x: mx.array, chunks: int, dim: int = 0) -> list[mx.array]:
    size = -(-x.shape[dim] // chunks)
    return mx.split(x, list(range(size, x.shape[dim], size)), axis=dim)


@_reg(aten.unbind.int)
def _unbind(x: mx.array, dim: int = 0) -> list[mx.array]:
    return [mx.take(x, i, axis=dim) for i in range(x.shape[dim])]


@_reg(aten._to_copy.default, aten.to.dtype, aten.to.dtype_layout)
def _to(x: mx.array, dtype: torch.dtype | None = None, **_: Any) -> mx.array:  # noqa: ANN401
    return x if dtype is None else x.astype(_dt(dtype))


OPS[aten.type_as.default] = lambda x, o: x.astype(o.dtype)


@_reg(aten.arange.default, aten.arange.start, aten.arange.start_step)
def _arange(*args: Any, dtype: torch.dtype | None = None, **_: Any) -> mx.array:  # noqa: ANN401
    out = mx.arange(*args)
    return out.astype(_dt(dtype)) if dtype is not None else out


@_reg(aten.full.default)
def _full(size: Sequence[int], value: Any, dtype: torch.dtype | None = None, **_: Any) -> mx.array:  # noqa: ANN401
    return mx.full(list(size), value, dtype=_dt(dtype))


@_reg(aten.zeros.default)
def _zeros(size: Sequence[int], dtype: torch.dtype | None = None, **_: Any) -> mx.array:  # noqa: ANN401
    return mx.zeros(list(size), dtype=_dt(dtype) or mx.float32)


@_reg(aten.ones.default)
def _ones(size: Sequence[int], dtype: torch.dtype | None = None, **_: Any) -> mx.array:  # noqa: ANN401
    return mx.ones(list(size), dtype=_dt(dtype) or mx.float32)


OPS[aten.full_like.default] = lambda x, v, **_: mx.full(x.shape, v, dtype=x.dtype)
OPS[aten.zeros_like.default] = lambda x, **_: mx.zeros_like(x)
OPS[aten.ones_like.default] = lambda x, **_: mx.ones_like(x)
OPS[aten.scalar_tensor.default] = lambda v, **_: mx.array(v)


@_reg(aten.index.Tensor)
def _index(x: mx.array, indices: Sequence[mx.array | None]) -> mx.array:
    idx = tuple(slice(None) if i is None else i for i in indices)
    return x[idx]


OPS[aten.embedding.default] = lambda w, i, *_: mx.take(w, i, axis=0)
OPS[aten.gather.default] = lambda x, d, i, *_: mx.take_along_axis(x, i, axis=d)
OPS[aten.index_select.default] = lambda x, d, i: mx.take(x, i, axis=d)


# ---- reductions -------------------------------------------------------------------------------------------------
OPS[aten.mean.dim] = lambda x, d=None, keep=False, **_: mx.mean(x, axis=tuple(d) if d else None, keepdims=keep)
OPS[aten.sum.dim_IntList] = lambda x, d=None, keep=False, **_: mx.sum(x, axis=tuple(d) if d else None, keepdims=keep)
OPS[aten.amax.default] = lambda x, d=(), keep=False: mx.max(x, axis=tuple(d) or None, keepdims=keep)
OPS[aten.amin.default] = lambda x, d=(), keep=False: mx.min(x, axis=tuple(d) or None, keepdims=keep)
OPS[aten.any.dim] = lambda x, d, keep=False: mx.any(x, axis=d, keepdims=keep)
OPS[aten.all.dim] = lambda x, d, keep=False: mx.all(x, axis=d, keepdims=keep)
OPS[aten.argmax.default] = lambda x, d=None, keep=False: mx.argmax(x, axis=d, keepdims=keep)
OPS[aten.argmin.default] = lambda x, d=None, keep=False: mx.argmin(x, axis=d, keepdims=keep)
OPS[aten.var.correction] = lambda x, d=None, correction=1, keepdim=False: mx.var(
    x, axis=tuple(d) if d else None, keepdims=keepdim, ddof=int(correction or 0)
)
OPS[aten.max.dim] = lambda x, d, keep=False: (mx.max(x, axis=d, keepdims=keep), mx.argmax(x, axis=d, keepdims=keep))
OPS[aten.max.default] = lambda x: mx.max(x)
OPS[aten.min.default] = lambda x: mx.min(x)


@_reg(aten.topk.default)
def _topk(x: mx.array, k: int, dim: int = -1, largest: bool = True, sorted_: bool = True) -> tuple[mx.array, mx.array]:  # noqa: FBT001, FBT002
    del sorted_
    order = mx.argsort(-x if largest else x, axis=dim)
    idx = mx.take(order, mx.arange(k), axis=dim)
    return mx.take_along_axis(x, idx, axis=dim), idx


# ---- linear algebra ---------------------------------------------------------------------------------------------
OPS[aten.mm.default] = OPS[aten.bmm.default] = OPS[aten.matmul.default] = mx.matmul
OPS[aten.addmm.default] = lambda b, x, w, beta=1, alpha=1: mx.addmm(b, x, w, alpha=alpha, beta=beta)
OPS[aten.baddbmm.default] = lambda b, x, w, beta=1, alpha=1: beta * b + alpha * (x @ w)
OPS[aten.dot.default] = mx.inner
OPS[aten.einsum.default] = lambda eq, ts: mx.einsum(eq, *ts)


@_reg(aten.scaled_dot_product_attention.default)
def _sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
    dropout: float = 0.0,
    causal: bool = False,  # noqa: FBT001, FBT002
    scale: float | None = None,
    enable_gqa: bool = False,  # noqa: FBT001, FBT002
) -> mx.array:
    del dropout, enable_gqa
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    if causal:
        mask = "causal"
    elif mask is not None and mask.dtype == mx.bool_:
        pass
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


@_reg(aten._softmax.default, aten.softmax.int)
def _softmax(x: mx.array, dim: int, half_to_float: bool = False) -> mx.array:  # noqa: FBT001, FBT002
    del half_to_float
    return mx.softmax(x, axis=dim, precise=True)


OPS[aten._log_softmax.default] = lambda x, d, h=False: x - mx.logsumexp(x.astype(mx.float32), axis=d, keepdims=True).astype(
    x.dtype
)


# ---- normalisation ----------------------------------------------------------------------------------------------
@_reg(aten.native_layer_norm.default)
def _layer_norm(
    x: mx.array, shape: Sequence[int], w: mx.array | None, b: mx.array | None, eps: float
) -> tuple[mx.array, None, None]:
    if len(shape) == 1:
        return mx.fast.layer_norm(x, w, b, eps), None, None
    axes = tuple(range(x.ndim - len(shape), x.ndim))
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=axes, keepdims=True)
    var = mx.var(xf, axis=axes, keepdims=True)
    y = ((xf - mean) * mx.rsqrt(var + eps)).astype(x.dtype)
    if w is not None:
        y = y * w
    if b is not None:
        y = y + b
    return y, None, None


@_reg(aten.native_group_norm.default)
def _group_norm(
    x: mx.array, w: mx.array | None, b: mx.array | None, n: int, c: int, hw: int, groups: int, eps: float
) -> tuple[mx.array, None, None]:
    del n, hw
    shape = x.shape
    xf = x.astype(mx.float32).reshape(shape[0], groups, -1)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.var(xf, axis=-1, keepdims=True)
    y = ((xf - mean) * mx.rsqrt(var + eps)).reshape(shape).astype(x.dtype)
    bshape = [1, c] + [1] * (len(shape) - 2)
    if w is not None:
        y = y * w.reshape(bshape)
    if b is not None:
        y = y + b.reshape(bshape)
    return y, None, None


@_reg(aten._native_batch_norm_legit_no_training.default, aten.native_batch_norm.default)
def _batch_norm(
    x: mx.array, w: mx.array | None, b: mx.array | None, mean: mx.array, var: mx.array, *rest: Any  # noqa: ANN401
) -> tuple[mx.array, None, None]:
    eps = rest[-1]
    shape = [1, -1] + [1] * (x.ndim - 2)
    scale = mx.rsqrt(var.astype(mx.float32) + eps)
    if w is not None:
        scale = scale * w.astype(mx.float32)
    shift = -mean.astype(mx.float32) * scale + (b.astype(mx.float32) if b is not None else 0)
    return x * scale.astype(x.dtype).reshape(shape) + shift.astype(x.dtype).reshape(shape), None, None


# ---- spatial ----------------------------------------------------------------------------------------------------
def _nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


@_reg(aten.convolution.default)
def _conv(
    x: mx.array,
    w: mx.array,
    b: mx.array | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,  # noqa: FBT001
    output_padding: Sequence[int],
    groups: int,
) -> mx.array:
    if x.ndim == 3:  # 1-D: [N, C, L] -> NLC
        xl = mx.transpose(x, (0, 2, 1))
        wl = mx.transpose(w, (0, 2, 1))
        if transposed:
            y = mx.conv_transpose1d(xl, mx.transpose(w, (1, 2, 0)), stride=stride[0], padding=padding[0],
                                    dilation=dilation[0], output_padding=output_padding[0], groups=groups)
        else:
            y = mx.conv1d(xl, wl, stride=stride[0], padding=padding[0], dilation=dilation[0], groups=groups)
        if b is not None:
            y = y + b
        return mx.transpose(y, (0, 2, 1))
    xl = _nhwc(x)
    if transposed:
        wt = mx.transpose(w, (1, 2, 3, 0))  # [Cin, Cout/g, kh, kw] -> [Cout/g, kh, kw, Cin]
        y = mx.conv_transpose2d(xl, wt, stride=tuple(stride), padding=tuple(padding), dilation=tuple(dilation),
                                output_padding=tuple(output_padding), groups=groups)
    else:
        wl = mx.transpose(w, (0, 2, 3, 1))
        y = mx.conv2d(xl, wl, stride=tuple(stride), padding=tuple(padding), dilation=tuple(dilation), groups=groups)
    if b is not None:
        y = y + b
    return _nchw(y)


def _pool_windows(
    x: mx.array, k: Sequence[int], s: Sequence[int], pad: Sequence[int], fill: float, *, ceil: bool = False
) -> mx.array:
    """NCHW -> strided windows `[N, C, OH, OW, kh, kw]` of the padded input."""
    extra = [0, 0]
    if ceil:
        for i in range(2):
            size = x.shape[2 + i] + 2 * pad[i] - k[i]
            out = -(-size // s[i]) + 1
            if (out - 1) * s[i] >= x.shape[2 + i] + pad[i]:  # the last window must start inside the input
                out -= 1
            extra[i] = max(0, (out - 1) * s[i] + k[i] - (x.shape[2 + i] + 2 * pad[i]))
    if any(pad) or any(extra):
        x = mx.pad(x, [(0, 0), (0, 0), (pad[0], pad[0] + extra[0]), (pad[1], pad[1] + extra[1])], constant_values=fill)
    n, c, h, w = x.shape
    oh, ow = (h - k[0]) // s[0] + 1, (w - k[1]) // s[1] + 1
    st = (c * h * w, h * w, s[0] * w, s[1], w, 1)
    return mx.as_strided(x, (n, c, oh, ow, k[0], k[1]), st)


@_reg(aten.max_pool2d_with_indices.default, aten.max_pool2d.default)
def _max_pool(
    x: mx.array, k: Sequence[int], s: Sequence[int] = (), pad: Sequence[int] = (0, 0), d: Sequence[int] = (1, 1),
    ceil: bool = False,  # noqa: FBT001, FBT002
) -> tuple[mx.array, None]:
    if tuple(d) != (1, 1):
        msg = "max_pool2d with dilation"
        raise UnsupportedOp(msg)
    s = s or k
    return _pool_windows(x, k, s, pad, -float("inf"), ceil=ceil).max(axis=(-2, -1)), None


@_reg(aten.avg_pool2d.default)
def _avg_pool(
    x: mx.array, k: Sequence[int], s: Sequence[int] = (), pad: Sequence[int] = (0, 0), ceil: bool = False,  # noqa: FBT001, FBT002
    count_include_pad: bool = True,  # noqa: FBT001, FBT002
    divisor: int | None = None,
) -> mx.array:
    if ceil or divisor is not None:
        msg = "avg_pool2d with ceil_mode or divisor_override"
        raise UnsupportedOp(msg)
    s = s or k
    win = _pool_windows(x, k, s, pad, 0.0).astype(mx.float32).sum(axis=(-2, -1))
    if any(pad) and not count_include_pad:
        ones = mx.ones((1, 1, *x.shape[2:]), mx.float32)
        win = win / _pool_windows(ones, k, s, pad, 0.0).sum(axis=(-2, -1))
    else:
        win = win / (k[0] * k[1])
    return win.astype(x.dtype)


@_reg(aten._adaptive_avg_pool2d.default, aten.adaptive_avg_pool2d.default)
def _adaptive_avg_pool(x: mx.array, size: Sequence[int]) -> mx.array:
    n, c, h, w = x.shape
    oh, ow = size
    if (oh, ow) == (1, 1):
        return mx.mean(x.astype(mx.float32), axis=(2, 3), keepdims=True).astype(x.dtype)
    if h % oh == 0 and w % ow == 0:
        y = x.astype(mx.float32).reshape(n, c, oh, h // oh, ow, w // ow).mean(axis=(3, 5))
        return y.astype(x.dtype)
    rows = []
    for i in range(oh):
        h0, h1 = (i * h) // oh, -((-(i + 1) * h) // oh)
        cols = []
        for j in range(ow):
            w0, w1 = (j * w) // ow, -((-(j + 1) * w) // ow)
            cols.append(x[:, :, h0:h1, w0:w1].astype(mx.float32).mean(axis=(2, 3)))
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=-2).astype(x.dtype)


@_reg(aten.constant_pad_nd.default)
def _pad(x: mx.array, pad: Sequence[int], value: float = 0.0) -> mx.array:
    widths = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):
        widths[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    return mx.pad(x, widths, constant_values=value)


@_reg(aten.reflection_pad2d.default, aten.replication_pad2d.default)
def _pad_edge(x: mx.array, pad: Sequence[int]) -> mx.array:
    msg = "reflection / replication padding"
    raise UnsupportedOp(msg)


def _src_index(out: int, inp: int, align: bool, scale: float | None) -> mx.array:  # noqa: FBT001
    i = mx.arange(out, dtype=mx.float32)
    if align:
        return i * ((inp - 1) / (out - 1)) if out > 1 else mx.zeros_like(i)
    ratio = (1.0 / scale) if scale else inp / out
    return mx.maximum((i + 0.5) * ratio - 0.5, 0.0)


@_reg(aten.upsample_bilinear2d.default, aten.upsample_bilinear2d.vec)
def _bilinear(x: mx.array, size: Sequence[int] | None, align: bool, *scales: Any) -> mx.array:  # noqa: ANN401, FBT001
    n, c, h, w = x.shape
    sh = sw = None
    if size is None:
        fac = scales[0]
        sh, sw = (fac[0], fac[1]) if isinstance(fac, (list, tuple)) else (scales[0], scales[1])
        oh, ow = int(h * sh), int(w * sw)
    else:
        oh, ow = size
        if len(scales) >= 2 and not isinstance(scales[0], (list, tuple)):
            sh, sw = scales[0], scales[1]
    ys, xs = _src_index(oh, h, align, sh), _src_index(ow, w, align, sw)
    y0 = mx.minimum(mx.floor(ys).astype(mx.int32), h - 1)
    x0 = mx.minimum(mx.floor(xs).astype(mx.int32), w - 1)
    y1, x1 = mx.minimum(y0 + 1, h - 1), mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0.astype(mx.float32)).astype(x.dtype)[:, None]
    wx = (xs - x0.astype(mx.float32)).astype(x.dtype)
    rows0, rows1 = mx.take(x, y0, axis=2), mx.take(x, y1, axis=2)
    top = mx.take(rows0, x0, axis=3) * (1 - wx) + mx.take(rows0, x1, axis=3) * wx
    bot = mx.take(rows1, x0, axis=3) * (1 - wx) + mx.take(rows1, x1, axis=3) * wx
    return top * (1 - wy) + bot * wy


@_reg(aten.upsample_nearest2d.default, aten.upsample_nearest2d.vec)
def _nearest(x: mx.array, size: Sequence[int] | None, *scales: Any) -> mx.array:  # noqa: ANN401
    h, w = x.shape[2:]
    if size is None:
        fac = scales[0]
        sh, sw = (fac[0], fac[1]) if isinstance(fac, (list, tuple)) else (scales[0], scales[1])
        oh, ow = int(h * sh), int(w * sw)
    else:
        oh, ow = size
    ys = mx.minimum(mx.floor(mx.arange(oh, dtype=mx.float32) * (h / oh)).astype(mx.int32), h - 1)
    xs = mx.minimum(mx.floor(mx.arange(ow, dtype=mx.float32) * (w / ow)).astype(mx.int32), w - 1)
    return mx.take(mx.take(x, ys, axis=2), xs, axis=3)


# ---- interpreter ------------------------------------------------------------------------------------------------
def _tensor_dtype(dtype: mx.Dtype, t: torch.Tensor) -> mx.Dtype | None:
    return dtype if t.is_floating_point() else None


class GraphNet:
    """An eval-mode `torch.nn.Module` run on MLX: `net(*tensors) -> tuple of float32 CPU tensors`.

    Exports once per input-shape signature (the exported view / expand sizes are static), shares the converted
    parameters between shapes, and compiles the replay with `mx.compile`.

    Args:
        module: The network in eval mode on the CPU. It is not kept; its parameters are converted and released.
        dtype: `"float32"` or `"float16"` (or `"bfloat16"`): the dtype floating parameters and activations run in.
        compile: `mx.compile` the replay (default).
    """

    def __init__(
        self, module: torch.nn.Module, dtype: str = "float32", *, compile: bool = True, kernels: bool = False  # noqa: A002
    ) -> None:
        """Keep the module until its first export; parameters convert then."""
        self.module = module.eval()
        self.dtype = getattr(mx, dtype)
        self._compile = compile
        self.kernels = kernels
        self._runs: dict[tuple[Any, ...], Callable[..., Any]] = {}
        self._consts: dict[str, mx.array] = {}

    def _const(self, name: str, t: torch.Tensor) -> mx.array:
        if name not in self._consts:
            a = to_mx(t)
            if t.is_floating_point():
                a = a.astype(self.dtype)
            mx.eval(a)
            self._consts[name] = a
        return self._consts[name]

    def _build(self, args: tuple[torch.Tensor, ...]) -> Callable[..., Any]:
        with torch.no_grad():
            ep = torch.export.export(self.module, args, strict=False)
            decomp = torch.export.default_decompositions()
            for keep in (aten.scaled_dot_product_attention.default, aten.upsample_bilinear2d.vec):
                decomp.pop(keep, None)
            ep = ep.run_decompositions(decomp)
        graph = ep.graph_module.graph
        sig = ep.graph_signature
        lifted = {**ep.state_dict, **dict(ep.constants)}
        names = {s.arg.name: s.target for s in sig.input_specs if s.target is not None}
        nodes = list(graph.nodes)
        plan: list[tuple[Any, ...]] = []
        values: dict[str, Any] = {}  # constants (parameters, buffers) by node name
        folds = self._fold_batch_norms(nodes, names, lifted)
        for node in nodes:
            if node.op == "placeholder":
                tgt = names.get(node.name)
                if tgt is not None:
                    values[node.name] = self._const(str(tgt), lifted[tgt])
                else:
                    plan.append(("input", node.name))
            elif node.op == "call_function":
                if node.name in folds.skip:
                    plan.append(("alias", node.name, folds.skip[node.name]))
                    continue
                args = folds.args.get(node.name, node.args)
                plan.append(("call", node.name, self._op(node.target), args, node.kwargs))
            elif node.op == "output":
                plan.append(("output", node.args[0]))
        values.update(folds.values)
        return self._runner(plan, values)

    @staticmethod
    def _op(target: Any) -> Callable[..., Any]:  # noqa: ANN401
        if target is operator.getitem:
            return operator.getitem
        try:
            return OPS[target]
        except KeyError:
            msg = f"no MLX mapping for {target}"
            raise UnsupportedOp(msg) from None

    def _fold_batch_norms(self, nodes: list[Any], names: dict[str, Any], lifted: dict[str, torch.Tensor]) -> Any:  # noqa: ANN401
        """Fold eval BatchNorm nodes that follow a convolution with constant weights into that convolution."""
        res = type("Folds", (), {"skip": {}, "values": {}, "args": {}})()
        bn_targets = (aten._native_batch_norm_legit_no_training.default,)
        for bn in nodes:
            if bn.op != "call_function" or bn.target not in bn_targets:
                continue
            users = [u for u in bn.users if u.target is operator.getitem]
            conv = bn.args[0]
            if (
                getattr(conv, "target", None) is not aten.convolution.default
                or len(conv.users) != 1
                or len(users) != 1
                or users[0].args[1] != 0
                or len(users[0].users) == 0
                or conv.args[6]  # transposed
            ):
                continue
            wname = names.get(conv.args[1].name)
            params = [names.get(a.name) if a is not None else None for a in bn.args[1:5]]
            if wname is None or params[2] is None or params[3] is None:
                continue
            eps = bn.args[-1]
            w = lifted[wname].double()
            gamma = lifted[params[0]].double() if params[0] else torch.ones(w.shape[0], dtype=torch.double)
            beta = lifted[params[1]].double() if params[1] else torch.zeros(w.shape[0], dtype=torch.double)
            scale = gamma / torch.sqrt(lifted[params[3]].double() + eps)
            bias0 = lifted[names[conv.args[2].name]].double() if conv.args[2] is not None else 0
            w2 = (w * scale.view(-1, *[1] * (w.ndim - 1))).float()
            b2 = ((bias0 - lifted[params[2]].double()) * scale + beta).float()
            wk, bk = f"{conv.name}__w", f"{conv.name}__b"
            res.values[wk] = to_mx(w2).astype(self.dtype)
            res.values[bk] = to_mx(b2).astype(self.dtype)
            mx.eval(res.values[wk], res.values[bk])
            # the convolution reads the folded constants; the bn node and its getitem alias the convolution's output
            res.args[conv.name] = (conv.args[0], _Name(wk), _Name(bk), *conv.args[3:])
            res.skip[bn.name] = conv.name
            res.skip[users[0].name] = conv.name
        return res

    def _runner(self, plan: list[tuple[Any, ...]], values: dict[str, Any]) -> Callable[..., Any]:
        out_names: Any = None
        for p in plan:
            if p[0] == "output":
                out_names = p[1]

        def run(*inputs: mx.array) -> tuple[mx.array, ...]:
            env: dict[str, Any] = dict(values)
            it = iter(inputs)

            def get(a: Any) -> Any:  # noqa: ANN401
                if isinstance(a, (_Name,)) or hasattr(a, "name") and hasattr(a, "op"):
                    return env[a.name]
                if isinstance(a, (list, tuple)):
                    return type(a)(get(x) for x in a)
                if isinstance(a, torch.dtype):
                    return a
                return a

            for p in plan:
                kind = p[0]
                if kind == "input":
                    x = next(it)
                    env[p[1]] = x.astype(self.dtype) if mx.issubdtype(x.dtype, mx.floating) else x
                elif kind == "alias":
                    env[p[1]] = env[p[2]]
                elif kind == "call":
                    _, name, fn, args, kwargs = p
                    if fn is operator.getitem:
                        env[name] = env[args[0].name][args[1]]
                    else:
                        env[name] = fn(*[get(a) for a in args], **{k: get(v) for k, v in kwargs.items()})
            out = get(out_names)
            return tuple(o.astype(mx.float32) for o in out)

        return mx.compile(run) if self._compile else run

    def __call__(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run on CPU tensors; returns float32 CPU tensors."""
        key = tuple((tuple(t.shape), str(t.dtype)) for t in tensors)
        if key not in self._runs:
            self._runs[key] = self._build(tuple(tensors))
        outs = self._runs[key](*[to_mx(t) for t in tensors])
        return tuple(to_torch(o) for o in outs)


class _Name:
    """A graph value by name (a folded constant)."""

    def __init__(self, name: str) -> None:
        self.name = name
