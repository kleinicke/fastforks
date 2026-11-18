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
# ruff: noqa: SLF001, ARG005, PLR0913, RUF015  # a table of aten overloads: private aten names, uniform lambda signatures

from __future__ import annotations

import gc
import math
import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.utils._pytree as pytree

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
OPS[aten.log1p.default] = mx.log1p
OPS[aten.expm1.default] = mx.expm1
OPS[aten.sqrt.default] = mx.sqrt
OPS[aten.rsqrt.default] = mx.rsqrt
OPS[aten.sin.default] = mx.sin
OPS[aten.cos.default] = mx.cos
OPS[aten.tanh.default] = mx.tanh
OPS[aten.acos.default] = mx.arccos
OPS[aten.asin.default] = mx.arcsin
OPS[aten.atan.default] = mx.arctan
OPS[aten.atan2.default] = mx.arctan2
OPS[aten.tan.default] = mx.tan
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
OPS[aten.softplus.default] = lambda x, beta=1.0, thr=20.0: mx.where(
    x * beta > thr, x, mx.log1p(mx.exp(x * beta)) / beta
)
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
OPS[aten.bitwise_and.Tensor] = mx.bitwise_and
OPS[aten.bitwise_or.Tensor] = mx.bitwise_or
OPS[aten.bitwise_xor.Tensor] = mx.bitwise_xor
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
OPS[aten.view.default] = OPS[aten._unsafe_view.default] = OPS[aten.reshape.default] = lambda x, s: mx.reshape(
    x, list(s)
)
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
OPS[aten.copy.default] = lambda dst, src, non_blocking=False: mx.broadcast_to(src.astype(dst.dtype), dst.shape)
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


@_reg(aten.slice_scatter.default)
def _slice_scatter(
    x: mx.array, src: mx.array, dim: int = 0, start: int | None = None, end: int | None = None, step: int = 1
) -> mx.array:
    n = x.shape[dim]
    start = 0 if start is None else max(0, start + n if start < 0 else start)
    end = n if end is None else min(n, end + n if end < 0 else end)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(start, max(start, end), step)
    y = mx.array(x)  # a new array: the assignment must not write through the input's graph node
    y[tuple(idx)] = src.astype(x.dtype)
    return y


@_reg(aten.div.Tensor_mode, aten.div.Scalar_mode)
def _div_mode(a: mx.array, b: mx.array | float, *, rounding_mode: str | None = None) -> mx.array:
    if rounding_mode is None:
        return a / b
    ints = not mx.issubdtype(mx.array(a).dtype, mx.floating) and not isinstance(b, float)
    if rounding_mode == "floor":
        return mx.floor_divide(a, b)
    q = mx.floor_divide(mx.abs(a), mx.abs(b)) if ints else mx.trunc(a / b)  # trunc: round toward zero
    return q * mx.sign(a) * mx.sign(b) if ints else q


@_reg(aten.grid_sampler_2d.default)
def _grid_sample(
    x: mx.array,
    grid: mx.array,
    mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,  # noqa: FBT001, FBT002
) -> mx.array:
    """Bilinear `grid_sample` (zeros or border padding): four gathers weighted in float32."""
    if mode != 0 or padding_mode not in (0, 1):
        msg = f"grid_sampler_2d mode={mode} padding_mode={padding_mode}"
        raise UnsupportedOp(msg)
    n, c, h, w = x.shape
    ho, wo = grid.shape[1:3]
    g = grid.astype(mx.float32)

    def unnormalize(v: mx.array, size: int) -> mx.array:
        v = (v + 1) * (size - 1) / 2 if align_corners else ((v + 1) * size - 1) / 2
        return mx.clip(v, 0, size - 1) if padding_mode == 1 else v

    ix, iy = unnormalize(g[..., 0], w).reshape(n, -1), unnormalize(g[..., 1], h).reshape(n, -1)
    x0, y0 = mx.floor(ix), mx.floor(iy)
    flat = x.reshape(n, c, h * w)
    out = None
    for dy, wy in ((0, 1 - (iy - y0)), (1, iy - y0)):
        for dx, wx in ((0, 1 - (ix - x0)), (1, ix - x0)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (xx <= w - 1) & (yy >= 0) & (yy <= h - 1)
            idx = (mx.clip(yy, 0, h - 1) * w + mx.clip(xx, 0, w - 1)).astype(mx.int32)
            v = mx.take_along_axis(flat, idx[:, None, :], axis=2).astype(mx.float32)
            term = v * (wx * wy * ok.astype(mx.float32))[:, None, :]
            out = term if out is None else out + term
    return out.reshape(n, c, ho, wo).astype(x.dtype)


@_reg(aten.scatter.src)
def _scatter_src(x: mx.array, dim: int, index: mx.array, src: mx.array) -> mx.array:
    src = src[tuple(slice(0, n) for n in index.shape)]  # torch reads the leading block of src
    wide = x.dtype in (mx.int64, mx.uint64)  # the GPU scatter has no 64-bit integers
    y = mx.put_along_axis(
        x.astype(mx.int32) if wide else x, index.astype(mx.int32), src.astype(mx.int32 if wide else x.dtype), axis=dim
    )
    return y.astype(x.dtype)


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
_KERNELS = (
    False  # kernels mode: our Metal kernels (`fastforks.kernels.graph_mlx`) for depthwise convolution and pooling
)


def _mean(x: mx.array, d: Sequence[int] | None = None, keep: bool = False, **_: object) -> mx.array:  # noqa: FBT001, FBT002
    if _KERNELS and not keep and x.ndim == 4 and d is not None and sorted(d) == [2, 3]:
        from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415 - optional MLX dependency

        return ops.global_avg(_nhwc(x))
    return _wide(x, lambda y: mx.mean(y, axis=tuple(d) if d else None, keepdims=keep))


def _wide(x: mx.array, f: Callable[[mx.array], mx.array]) -> mx.array:
    """`f(x)` accumulated in float32 for a half-precision `x` (PyTorch's reductions do; a float16 sum overflows)."""
    if x.dtype in (mx.float16, mx.bfloat16):
        return f(x.astype(mx.float32)).astype(x.dtype)
    return f(x)


OPS[aten.mean.dim] = _mean
OPS[aten.sum.dim_IntList] = lambda x, d=None, keep=False, **_: _wide(
    x, lambda y: mx.sum(y, axis=tuple(d) if d else None, keepdims=keep)
)


# FFTs (LaMa's Fourier units). aten's `normalization`: 0 none, 1 by_root_n (norm="ortho"), 2 by_n (inverse default).
def _fft_scale(normalization: int, n: int) -> float:
    return (1.0, n**-0.5, 1.0 / n)[normalization]


@_reg(aten._fft_r2c.default)
def _fft_r2c(x: mx.array, dim: Sequence[int], normalization: int, onesided: bool) -> mx.array:  # noqa: FBT001
    axes = tuple(dim)
    n = math.prod(x.shape[a] for a in axes)
    y = mx.fft.rfftn(x.astype(mx.float32), axes=axes)
    if not onesided:
        msg = "aten._fft_r2c with onesided=False"
        raise UnsupportedOp(msg)
    return y * _fft_scale(normalization, n)


@_reg(aten._fft_c2r.default)
def _fft_c2r(x: mx.array, dim: Sequence[int], normalization: int, last_dim_size: int) -> mx.array:
    axes = tuple(dim)
    sizes = [x.shape[a] for a in axes]
    sizes[-1] = last_dim_size
    n = math.prod(sizes)
    y = mx.fft.irfftn(x, s=sizes, axes=axes)  # normalised by 1/n
    return y * (n * _fft_scale(normalization, n))


OPS[aten.empty.memory_format] = lambda size, dtype=None, **_: mx.zeros(tuple(size), dtype=_dt(dtype) or mx.float32)
OPS[aten.view_as_real.default] = lambda x: mx.stack([mx.real(x), mx.imag(x)], axis=-1)
OPS[aten.view_as_complex.default] = lambda x: x[..., 0] + 1j * x[..., 1]


@_reg(aten.linalg_vector_norm.default)
def _vector_norm(
    x: mx.array,
    ord: float = 2,  # noqa: A002 - the aten argument name
    d: Sequence[int] | None = None,
    keep: bool = False,  # noqa: FBT001, FBT002
    *,
    dtype: object = None,  # noqa: ARG001 - the aten argument
) -> mx.array:
    """The p-norm with float32 accumulation (a float16 sum of squares overflows; PyTorch accumulates in float32)."""
    y = mx.abs(x.astype(mx.float32))
    ax = tuple(d) if d else None
    if ord == float("inf"):
        r = mx.max(y, axis=ax, keepdims=keep)
    elif ord == 2:  # noqa: PLR2004
        r = mx.sqrt(mx.sum(y * y, axis=ax, keepdims=keep))
    elif ord == 1:
        r = mx.sum(y, axis=ax, keepdims=keep)
    else:
        r = mx.power(mx.sum(mx.power(y, ord), axis=ax, keepdims=keep), 1.0 / ord)
    return r.astype(x.dtype)


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


@_reg(aten.sort.default, aten.sort.stable)
def _sort(x: mx.array, *args: Any, **kw: Any) -> tuple[mx.array, mx.array]:  # noqa: ANN401
    dim = kw.get("dim", -1)
    descending = kw.get("descending", False)
    pos = [a for a in args if not isinstance(a, bool)]
    if pos:
        dim = pos[0]
    bools = [a for a in args if isinstance(a, bool)]
    if bools:
        descending = bools[-1]
    order = mx.argsort(-x if descending else x, axis=dim)
    return mx.take_along_axis(x, order, axis=dim), order


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
    elif mask is not None and mask.dtype not in {mx.bool_, q.dtype}:
        mask = mask.astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


@_reg(aten._softmax.default, aten.softmax.int)
def _softmax(x: mx.array, dim: int, half_to_float: bool = False) -> mx.array:  # noqa: FBT001, FBT002
    del half_to_float
    return mx.softmax(x, axis=dim, precise=True)


OPS[aten._log_softmax.default] = lambda x, d, h=False: (
    x - mx.logsumexp(x.astype(mx.float32), axis=d, keepdims=True).astype(x.dtype)
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
    x: mx.array,
    w: mx.array | None,
    b: mx.array | None,
    mean: mx.array,
    var: mx.array,
    *rest: Any,  # noqa: ANN401
) -> tuple[mx.array, None, None]:
    eps = rest[-1]
    shape = [1, -1] + [1] * (x.ndim - 2)
    scale = mx.rsqrt(var.astype(mx.float32) + eps)
    if w is not None:
        scale = scale * w.astype(mx.float32)
    shift = -mean.astype(mx.float32) * scale + (b.astype(mx.float32) if b is not None else 0)
    return x * scale.astype(x.dtype).reshape(shape) + shift.astype(x.dtype).reshape(shape), None, None


@_reg(aten._native_batch_norm_legit.no_stats)
def _batch_norm_batch_stats(
    x: mx.array,
    w: mx.array | None,
    b: mx.array | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[mx.array, None, None]:
    """Normalisation by the batch's own per-channel statistics (what `InstanceNorm2d` decomposes to)."""
    del momentum
    if not training:
        msg = "batch norm without statistics outside training"
        raise UnsupportedOp(msg)
    axes = tuple(i for i in range(x.ndim) if i != 1)
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=axes, keepdims=True)
    var = mx.var(xf, axis=axes, keepdims=True)
    y = ((xf - mean) * mx.rsqrt(var + eps)).astype(x.dtype)
    shape = [1, -1] + [1] * (x.ndim - 2)
    if w is not None:
        y = y * w.reshape(shape)
    if b is not None:
        y = y + b.reshape(shape)
    return y, None, None


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
            y = mx.conv_transpose1d(
                xl,
                mx.transpose(w, (1, 2, 0)),
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
                output_padding=output_padding[0],
                groups=groups,
            )
        else:
            y = mx.conv1d(xl, wl, stride=stride[0], padding=padding[0], dilation=dilation[0], groups=groups)
        if b is not None:
            y = y + b
        return mx.transpose(y, (0, 2, 1))
    xl = _nhwc(x)
    if transposed:
        wt = mx.transpose(w, (1, 2, 3, 0))  # [Cin, Cout/g, kh, kw] -> [Cout/g, kh, kw, Cin]
        y = mx.conv_transpose2d(
            xl,
            wt,
            stride=tuple(stride),
            padding=tuple(padding),
            dilation=tuple(dilation),
            output_padding=tuple(output_padding),
            groups=groups,
        )
    else:
        wl = mx.transpose(w, (0, 2, 3, 1))
        if (
            _KERNELS
            and groups > 1
            and groups == xl.shape[3] == w.shape[0]
            and groups % 4 == 0
            and w.shape[2] * w.shape[3] > 1
        ):
            from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415 - optional MLX dependency

            bias = b if b is not None else mx.zeros((groups,), dtype=xl.dtype)
            return _nchw(
                ops.depthwise(
                    xl, mx.transpose(w[:, 0], (1, 2, 0)), bias, tuple(stride), tuple(padding), tuple(dilation)
                )
            )
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
    x: mx.array,
    k: Sequence[int],
    s: Sequence[int] = (),
    pad: Sequence[int] = (0, 0),
    d: Sequence[int] = (1, 1),
    ceil: bool = False,  # noqa: FBT001, FBT002
) -> tuple[mx.array, None]:
    if tuple(d) != (1, 1):
        msg = "max_pool2d with dilation"
        raise UnsupportedOp(msg)
    s = s or k
    return _pool_windows(x, k, s, pad, -float("inf"), ceil=ceil).max(axis=(-2, -1)), None


@_reg(aten.avg_pool2d.default)
def _avg_pool(
    x: mx.array,
    k: Sequence[int],
    s: Sequence[int] = (),
    pad: Sequence[int] = (0, 0),
    ceil: bool = False,  # noqa: FBT001, FBT002
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


def _aa_matrix(inp: int, out: int, align: bool, scale: float | None) -> np.ndarray:  # noqa: FBT001
    """`[out, inp]` weights of PyTorch's antialiased bilinear resize along one axis (triangle filter, normalised)."""
    s = (inp - 1) / (out - 1) if align and out > 1 else (1.0 / scale if scale else inp / out)
    support = max(s, 1.0)
    inv = 1.0 / s if s >= 1.0 else 1.0
    m = np.zeros((out, inp), dtype=np.float32)
    for i in range(out):
        c = s * i if align else s * (i + 0.5)
        lo = max(int(c - support + 0.5), 0)
        hi = min(int(c + support + 0.5), inp)
        w = np.maximum(0.0, 1.0 - np.abs((np.arange(lo, hi) - c + (0.0 if align else 0.5)) * inv))
        m[i, lo:hi] = w / w.sum() if w.sum() > 0 else 0.0
    return m


@_reg(aten._upsample_bilinear2d_aa.default, aten._upsample_bilinear2d_aa.vec)
def _bilinear_aa(x: mx.array, size: Sequence[int] | None, align: bool, *scales: Any) -> mx.array:  # noqa: ANN401, FBT001
    h, w = x.shape[2:]
    sh = sw = None
    if size is None:
        fac = scales[0]
        sh, sw = (fac[0], fac[1]) if isinstance(fac, (list, tuple)) else (scales[0], scales[1])
        oh, ow = int(h * sh), int(w * sw)
    else:
        oh, ow = size
        if len(scales) >= 2 and not isinstance(scales[0], (list, tuple)):
            sh, sw = scales[0], scales[1]
    wy = mx.array(_aa_matrix(h, oh, align, sh)).astype(x.dtype)
    wx = mx.array(_aa_matrix(w, ow, align, sw)).astype(x.dtype)
    return mx.matmul(mx.matmul(wy, x), wx.T)


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
        self,
        module: torch.nn.Module,
        dtype: str = "float32",
        *,
        compile: bool = True,
        kernels: bool = False,
        shared: dict[str, Any] | None = None,
    ) -> None:
        """Keep the module until its first export; parameters convert then."""
        self.module = module.eval()
        for m in self.module.modules():  # a buffer computed from parameters (position biases) is a non-leaf tensor
            for name, b in m._buffers.items():
                if b is not None and b.requires_grad:
                    m._buffers[name] = b.detach()
        self.dtype = getattr(mx, dtype)
        self._compile = compile
        self.kernels = kernels
        self._runs: dict[tuple[Any, ...], Callable[..., Any]] = {}
        self._consts: dict[str, mx.array] = {} if shared is None else shared

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
            try:
                ep = torch.export.export(self.module, args, strict=False)
            except (RuntimeError, ValueError, AssertionError, UserWarning):
                # lazily built buffers (attention bias caches) that export must not assign: fill them with one real
                # forward (its CPU activations are the peak of the build, so only a module that needs it pays)
                self.module(*args)
                ep = torch.export.export(self.module, args, strict=False)
            decomp = torch.export.default_decompositions()
            for keep in (
                aten.scaled_dot_product_attention.default,
                aten.upsample_bilinear2d.vec,
                aten.linalg_vector_norm.default,
            ):
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
        del ep, graph, sig, lifted, nodes
        gc.collect()
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
                if isinstance(a, (_Name,)) or (hasattr(a, "name") and hasattr(a, "op")):
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
        global _KERNELS  # noqa: PLW0603 - read by the op table while tracing
        _KERNELS = self.kernels
        key = tuple((tuple(t.shape), str(t.dtype)) for t in tensors)
        if key not in self._runs:
            self._runs[key] = self._build(tuple(tensors))
        outs = self._runs[key](*[to_mx(t) for t in tensors])
        return tuple(to_torch(o) for o in outs)


class _Box:
    """A value cell that `copy.deepcopy` (which `torch.export` applies to a module) shares instead of copying."""

    def __init__(self) -> None:
        self.value: tuple[Any, list[Any], list[Any]] | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> _Box:
        return self


class _Flat(torch.nn.Module):
    """`module(*args, **kwargs)` as a function of its tensor arguments (the rest is baked in), with flat tensor outputs."""

    def __init__(self, module: torch.nn.Module, spec: Any, leaves: list[Any]) -> None:  # noqa: ANN401
        super().__init__()
        self.module = module
        self.spec = spec
        self.template = [None if isinstance(x, torch.Tensor) else x for x in leaves]
        self.is_tensor = [isinstance(x, torch.Tensor) for x in leaves]
        self.out = _Box()  # out spec, non-tensor leaves, tensor dtypes; set by forward, also on an exported copy

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        it = iter(tensors)
        leaves = [next(it) if t else v for t, v in zip(self.is_tensor, self.template, strict=True)]
        args, kwargs = pytree.tree_unflatten(leaves, self.spec)
        out_leaves, out_spec = pytree.tree_flatten(self.module(*args, **kwargs))
        self.out.value = (
            out_spec,
            [None if isinstance(o, torch.Tensor) else o for o in out_leaves],
            [o.dtype if isinstance(o, torch.Tensor) else None for o in out_leaves],
        )
        return tuple(o for o in out_leaves if isinstance(o, torch.Tensor))


class MlxModule:
    """A module with any tensor / python-value arguments and nested outputs, run on MLX (`GraphNet` underneath).

    `net(*args, **kwargs)` takes CPU tensors and returns the module's output structure with CPU tensors (dtype as the
    module returns them). Python values in the arguments are baked into the export (part of the cache key).
    """

    def __init__(self, module: torch.nn.Module, dtype: str = "float32", *, kernels: bool = False) -> None:
        """Wrap `module` (eval mode, CPU)."""
        self.module = module.eval()
        self.dtype = dtype
        self.kernels = kernels
        self._nets: dict[Any, tuple[GraphNet, _Flat]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the module on MLX."""
        leaves, spec = pytree.tree_flatten((args, kwargs))
        key = (spec, tuple((tuple(x.shape), x.dtype) if isinstance(x, torch.Tensor) else repr(x) for x in leaves))
        if key not in self._nets:
            flat = _Flat(self.module, spec, leaves)
            self._nets[key] = (GraphNet(flat, self.dtype, kernels=self.kernels, shared=self._consts()), flat)
        net, flat = self._nets[key]
        outs = iter(net(*[x for x in leaves if isinstance(x, torch.Tensor)]))
        assert flat.out.value is not None  # set while exporting  # noqa: S101
        out_spec, others, dtypes = flat.out.value
        merged = [next(outs).to(d) if d is not None else o for o, d in zip(others, dtypes, strict=True)]
        return pytree.tree_unflatten(merged, out_spec)

    def parameters(self) -> Any:  # noqa: ANN401
        """The wrapped module's (CPU float32) parameters, for runners that read the dtype from them."""
        return self.module.parameters()

    def _consts(self) -> dict[str, Any]:
        return self.__dict__.setdefault("_shared_consts", {})


class _Name:
    """A graph value by name (a folded constant)."""

    def __init__(self, name: str) -> None:
        self.name = name
