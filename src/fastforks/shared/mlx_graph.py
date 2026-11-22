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
# a table of aten overloads: private aten names, uniform lambda signatures

from __future__ import annotations

import contextlib
import ctypes
import functools
import gc
import itertools
import math
import mmap
import operator
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.utils._pytree as pytree

from fastforks._core import mlx_cast as _mlx_cast  # noqa: F401 - registers `fastforks::run_dtype` / `float32`
from fastforks._core import volume_ops as _volume_ops  # noqa: F401 - registers `fastforks::gwc_volume`
from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks._core.tracing import EXPORTING_FOR_MLX
from fastforks.shared import mlx_dtype_report, mlx_plan_cache

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

__all__ = ["GraphNet", "UnsupportedOp"]

aten = torch.ops.aten


class UnsupportedOp(NotImplementedError):
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
    torch.uint16: mx.uint16,
    torch.uint32: mx.uint32,
    torch.bool: mx.bool_,
    torch.float64: mx.float32,
}


def _dt(d: torch.dtype | None) -> mx.Dtype | None:
    return None if d is None else _DTYPES[d]


OPS: dict[Any, Callable[..., Any]] = {}


def _reg(*targets: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        for t in targets:
            OPS[t] = fn
        return fn

    return deco


def _reg_cl(*targets: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the channels-last form of an op (`CL_OPS`), used where its operands are channels-last views."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        for t in targets:
            CL_OPS[t] = fn
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
def _alpha(b: Any, alpha: Any) -> Any:
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
OPS[aten.elu.default] = lambda x, a=1.0, scale=1.0, input_scale=1.0: (
    scale * mx.where(x > 0, x, a * (mx.exp(x * input_scale) - 1))
)  # SELU exports as elu(x, alpha, scale)
OPS[aten.softplus.default] = lambda x, beta=1.0, thr=20.0: mx.where(
    x * beta > thr, x, mx.log1p(mx.exp(x * beta)) / beta
)
OPS[aten.pow.Tensor_Scalar] = lambda x, e: mx.power(x, e)
OPS[aten.pow.Tensor_Tensor] = mx.power
OPS[aten.pow.Scalar] = lambda b, e: mx.power(b, e)
OPS[aten.maximum.default] = mx.maximum
OPS[aten.minimum.default] = mx.minimum
OPS[aten.clamp.default] = lambda x, lo=None, hi=None: mx.clip(x, lo, hi)
OPS[aten.clamp.Tensor] = lambda x, lo=None, hi=None: mx.clip(x, lo, hi)
OPS[aten.clamp_min.Tensor] = lambda x, lo: mx.maximum(x, lo)
OPS[aten.clamp_max.Tensor] = lambda x, hi: mx.minimum(x, hi)
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


def _where(c: mx.array, a: Any, b: Any) -> mx.array:
    """`torch.where` with torch's promotion of a 0-dim float operand.

    An exported `scalar_tensor` (float32) takes the other float operand's dtype, as in torch, instead of lifting a
    half-precision replay to float32 (MLX's rule).
    """
    if isinstance(a, mx.array) and isinstance(b, mx.array) and a.dtype != b.dtype:
        if a.ndim == 0 and b.ndim and mx.issubdtype(a.dtype, mx.floating) and mx.issubdtype(b.dtype, mx.floating):
            a = a.astype(b.dtype)
        elif b.ndim == 0 and a.ndim and mx.issubdtype(a.dtype, mx.floating) and mx.issubdtype(b.dtype, mx.floating):
            b = b.astype(a.dtype)
    return mx.where(c, a, b)


OPS[aten.where.self] = _where
OPS[aten.where.ScalarOther] = OPS[aten.where.ScalarSelf] = OPS[aten.where.self]
OPS[aten.masked_fill.Scalar] = lambda x, m, v: mx.where(m, mx.array(v, x.dtype), x)
OPS[aten.remainder.Scalar] = lambda a, b: mx.remainder(a, b)


def _fmod(a: Any, b: Any) -> Any:
    r = mx.remainder(a, b)
    return mx.where((r != 0) & ((a < 0) != (b < 0)), r - b, r)


OPS[aten.fmod.Scalar] = OPS[aten.fmod.Tensor] = _fmod
OPS[aten.floor_divide.default] = lambda a, b: mx.floor_divide(a, b)


def _gelu_tanh(x: mx.array) -> mx.array:
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + mx.tanh(c * (x + 0.044715 * x * x * x)))


def _gelu_erf(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))


if not os.environ.get("FF_MLX_NO_GELU_COMPILE"):  # one fused kernel even in an eager segment (5 passes otherwise)
    _gelu_tanh, _gelu_erf = mx.compile(_gelu_tanh, shapeless=True), mx.compile(_gelu_erf, shapeless=True)


@_reg(aten.gelu.default)
def _gelu(x: mx.array, approximate: str = "none") -> mx.array:
    return _gelu_tanh(x) if approximate == "tanh" else _gelu_erf(x)


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
OPS[aten.select.int] = lambda x, d, i: mx.take(
    x, i + x.shape[d] if i < 0 else i, axis=d
)  # (mx.take: no negative scalar)
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


def _roll(x: mx.array, shifts: Any, dims: Any) -> mx.array:
    if not dims:
        return mx.roll(x, next(iter(shifts)))
    return mx.roll(x, list(shifts), axis=list(dims))


@_reg(aten.expand.default)
def _expand(x: mx.array, size: Sequence[int], implicit: bool = False) -> mx.array:
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
    align_corners: bool = False,
) -> mx.array:
    """Bilinear `grid_sample` (zeros or border padding): four gathers weighted in float32."""
    if mode != 0 or padding_mode not in (0, 1):
        msg = f"grid_sampler_2d mode={mode} padding_mode={padding_mode}"
        raise UnsupportedOp(msg)
    n, c, h, w = x.shape
    ho, wo = grid.shape[1:3]
    if x.dtype in (mx.float16, mx.bfloat16, mx.float32) and not os.environ.get("FF_MLX_NO_FUSED_GRID"):
        from fastforks.kernels.graph_mlx import ops

        return _nchw(
            ops.grid_sample(_nhwc(x), grid.astype(mx.float32), align_corners=align_corners, border=padding_mode == 1)
        )
    g = grid.astype(mx.float32)

    def unnormalize(v: mx.array, size: int) -> mx.array:
        v = (v + 1) * (size - 1) / 2 if align_corners else ((v + 1) * size - 1) / 2
        return mx.clip(v, 0, size - 1) if padding_mode == 1 else v

    ix, iy = unnormalize(g[..., 0], w).reshape(n, -1), unnormalize(g[..., 1], h).reshape(n, -1)
    x0, y0 = mx.floor(ix), mx.floor(iy)
    flat = mx.transpose(x.reshape(n, c, h * w), (0, 2, 1))  # [N, HW, C]: a corner's channels are one contiguous row
    out = None
    for dy, wy in ((0, 1 - (iy - y0)), (1, iy - y0)):
        for dx, wx in ((0, 1 - (ix - x0)), (1, ix - x0)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (xx <= w - 1) & (yy >= 0) & (yy <= h - 1)
            idx = (mx.clip(yy, 0, h - 1) * w + mx.clip(xx, 0, w - 1)).astype(mx.int32)
            v = mx.take_along_axis(flat, idx[:, :, None], axis=1).astype(mx.float32)
            term = v * (wx * wy * ok.astype(mx.float32))[:, :, None]
            out = term if out is None else out + term
    out = mx.transpose(out, (0, 2, 1))
    return out.reshape(n, c, ho, wo).astype(x.dtype)


@_reg(aten.grid_sampler_3d.default)
def _grid_sample_3d(
    x: mx.array,
    grid: mx.array,
    mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> mx.array:
    """Trilinear `grid_sample` (zeros or border padding): eight gathers weighted in float32."""
    if mode != 0 or padding_mode not in (0, 1):
        msg = f"grid_sampler_3d mode={mode} padding_mode={padding_mode}"
        raise UnsupportedOp(msg)
    n, c, d, h, w = x.shape
    out_shape = grid.shape[1:4]
    g = grid.astype(mx.float32)

    def unnormalize(v: mx.array, size: int) -> mx.array:
        v = (v + 1) * (size - 1) / 2 if align_corners else ((v + 1) * size - 1) / 2
        return mx.clip(v, 0, size - 1) if padding_mode == 1 else v

    ix, iy, iz = (unnormalize(g[..., k], s).reshape(n, -1) for k, s in enumerate((w, h, d)))
    x0, y0, z0 = mx.floor(ix), mx.floor(iy), mx.floor(iz)
    flat = mx.transpose(x.reshape(n, c, d * h * w), (0, 2, 1))
    out = None
    for dz, wz in ((0, 1 - (iz - z0)), (1, iz - z0)):
        for dy, wy in ((0, 1 - (iy - y0)), (1, iy - y0)):
            for dx, wx in ((0, 1 - (ix - x0)), (1, ix - x0)):
                xx, yy, zz = x0 + dx, y0 + dy, z0 + dz
                ok = (xx >= 0) & (xx <= w - 1) & (yy >= 0) & (yy <= h - 1) & (zz >= 0) & (zz <= d - 1)
                idx = ((mx.clip(zz, 0, d - 1) * h + mx.clip(yy, 0, h - 1)) * w + mx.clip(xx, 0, w - 1)).astype(mx.int32)
                v = mx.take_along_axis(flat, idx[:, :, None], axis=1).astype(mx.float32)
                term = v * (wx * wy * wz * ok.astype(mx.float32))[:, :, None]
                out = term if out is None else out + term
    return mx.transpose(out, (0, 2, 1)).reshape(n, c, *out_shape).astype(x.dtype)


@_reg(torch.ops.fastforks.gwc_volume.default)
def _gwc_volume(left: mx.array, right: mx.array, depth: int, groups: int) -> mx.array:
    """The group-wise correlation volume op (`fastforks._core.volume_ops`): one Metal kernel, `[B, G, D, H, W]`."""
    from fastforks.kernels.graph_mlx import ops

    return mx.transpose(ops.gwc_volume(_nhwc(left), _nhwc(right), depth, groups), (0, 4, 1, 2, 3))


@_reg(torch.ops.fastforks.upsample_soft_argmax.default)
def _upsample_soft_argmax(cost: mx.array, max_disp: int, height: int, width: int) -> mx.array:
    """Trilinear upsample + softmax over disparity + soft-argmin (`fastforks._core.volume_ops`) as one Metal kernel."""
    from fastforks.kernels.graph_mlx import ops

    return ops.upsample_soft_argmax(cost, max_disp, height, width)


@_reg(aten.scatter.src)
def _scatter_src(x: mx.array, dim: int, index: mx.array, src: mx.array) -> mx.array:
    src = src[tuple(slice(0, n) for n in index.shape)]  # torch reads the leading block of src
    wide = x.dtype in (mx.int64, mx.uint64)  # the GPU scatter has no 64-bit integers
    y = mx.put_along_axis(
        x.astype(mx.int32) if wide else x, index.astype(mx.int32), src.astype(mx.int32 if wide else x.dtype), axis=dim
    )
    return y.astype(x.dtype)


@_reg(aten.scatter_add.default)
def _scatter_add(x: mx.array, dim: int, index: mx.array, src: mx.array) -> mx.array:
    """`x.scatter_add(dim, index, src)`: duplicate indices accumulate."""
    dim %= x.ndim
    src = src[tuple(slice(0, n) for n in index.shape)]  # torch reads the leading block of src
    idx = tuple(
        index.astype(mx.int32)
        if d == dim
        else mx.arange(index.shape[d], dtype=mx.int32).reshape([-1 if k == d else 1 for k in range(index.ndim)])
        for d in range(index.ndim)
    )
    return x.at[idx].add(src.astype(x.dtype))


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
def _to(x: mx.array, dtype: torch.dtype | None = None, **_: Any) -> mx.array:
    return x if dtype is None else x.astype(_dt(dtype))


OPS[aten.type_as.default] = lambda x, o: x.astype(o.dtype)


def _run_dtype(x: mx.array) -> mx.array:
    """`fastforks::run_dtype` (`fastforks._core.mlx_cast`): the plan binds the replay's dtype in its place."""
    return x


OPS[torch.ops.fastforks.run_dtype.default] = _run_dtype


def _float32(x: mx.array) -> mx.array:
    """`fastforks::float32` (`fastforks._core.mlx_cast.to_float32`): a float32 island of a float16 replay."""
    return x if x.dtype == mx.float32 else x.astype(mx.float32)


OPS[torch.ops.fastforks.float32.default] = _float32


@_reg(aten.arange.default, aten.arange.start, aten.arange.start_step)
def _arange(*args: Any, dtype: torch.dtype | None = None, **_: Any) -> mx.array:
    out = mx.arange(*args)
    return out.astype(_dt(dtype)) if dtype is not None else out


@_reg(aten.full.default)
def _full(size: Sequence[int], value: Any, dtype: torch.dtype | None = None, **_: Any) -> mx.array:
    return mx.full(list(size), value, dtype=_dt(dtype))


@_reg(aten.zeros.default)
def _zeros(size: Sequence[int], dtype: torch.dtype | None = None, **_: Any) -> mx.array:
    return mx.zeros(list(size), dtype=_dt(dtype) or mx.float32)


@_reg(aten.ones.default)
def _ones(size: Sequence[int], dtype: torch.dtype | None = None, **_: Any) -> mx.array:
    return mx.ones(list(size), dtype=_dt(dtype) or mx.float32)


OPS[aten.full_like.default] = lambda x, v, **_: mx.full(x.shape, v, dtype=x.dtype)
OPS[aten.zeros_like.default] = lambda x, **_: mx.zeros_like(x)
OPS[aten.ones_like.default] = lambda x, **_: mx.ones_like(x)
OPS[aten.scalar_tensor.default] = lambda v, **_: mx.array(v)


@_reg(aten.index.Tensor)
def _index(x: mx.array, indices: Sequence[mx.array | None]) -> mx.array:
    idx = tuple(slice(None) if i is None else i for i in indices)
    return x[idx]


@_reg(aten.index_put.default)
def _index_put(x: mx.array, indices: Sequence[mx.array | None], values: mx.array, accumulate: bool = False) -> mx.array:
    """`x[indices] = values`; a boolean mask with one value (or one per element) is a `where`."""
    if len(indices) == 1 and indices[0] is not None and indices[0].dtype == mx.bool_:
        mask = indices[0]
        mask = mask.reshape((*mask.shape, *([1] * (x.ndim - mask.ndim))))
        v = values.astype(x.dtype)
        if accumulate:
            v = x + v
        if values.size == 1:
            return mx.where(mask, v.reshape(()), x)
        return mx.where(mask, mx.broadcast_to(v, x.shape), x)
    idx = tuple(slice(None) if i is None else i for i in indices)
    values = values.astype(x.dtype)
    return x.at[idx].add(values) if accumulate else _set_at(x, idx, values)


def _set_at(x: mx.array, idx: tuple[object, ...], values: mx.array) -> mx.array:
    y = mx.array(x)
    y[idx] = values
    return y


OPS[aten.embedding.default] = lambda w, i, *_: mx.take(w, i, axis=0)
OPS[aten.gather.default] = lambda x, d, i, *_: mx.take_along_axis(x, i, axis=d)
OPS[aten.index_select.default] = lambda x, d, i: mx.take(x, i, axis=d)


# ---- reductions -------------------------------------------------------------------------------------------------
_KERNELS = (
    False  # kernels mode: our Metal kernels (`fastforks.kernels.graph_mlx`) for depthwise convolution and pooling
)
_HALF_WINO_F32: bool | tuple[tuple[int, int], ...] = False  # `GraphNet` option: half Winograd 3x3 convs run in float32
# The depthwise kernel in portable mode too: `mx.conv2d(groups=C)` falls off its fast path at some channel counts
# (DAT's 180-channel 3x3 at 512^2: 75 ms against 2.6 ms, 72 per call, 5.7 of 16.4 s) and is never faster.
DW_KERNEL = not os.environ.get("FF_MLX_NO_DW_KERNEL")


def _mean(x: mx.array, d: Sequence[int] | None = None, keep: bool = False, **_: object) -> mx.array:
    if _KERNELS and not keep and x.ndim == 4 and d is not None and sorted(d) == [2, 3]:
        from fastforks.kernels.graph_mlx import ops

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
def _fft_r2c(x: mx.array, dim: Sequence[int], normalization: int, onesided: bool) -> mx.array:
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
    ord: float = 2,
    d: Sequence[int] | None = None,
    keep: bool = False,
    *,
    dtype: object = None,
) -> mx.array:
    """The p-norm with float32 accumulation (a float16 sum of squares overflows; PyTorch accumulates in float32)."""
    y = mx.abs(x.astype(mx.float32))
    ax = tuple(d) if d else None
    if ord == float("inf"):
        r = mx.max(y, axis=ax, keepdims=keep)
    elif ord == 2:
        r = mx.sqrt(mx.sum(y * y, axis=ax, keepdims=keep))
    elif ord == 1:
        r = mx.sum(y, axis=ax, keepdims=keep)
    else:
        r = mx.power(mx.sum(mx.power(y, ord), axis=ax, keepdims=keep), 1.0 / ord)
    return r.astype(x.dtype)


OPS[aten.amax.default] = lambda x, d=(), keep=False: mx.max(x, axis=tuple(d) or None, keepdims=keep)
OPS[aten.prod.dim_int] = lambda x, d, keep=False, **_: _wide(x, lambda y: mx.prod(y, axis=d, keepdims=keep))
OPS[aten.amin.default] = lambda x, d=(), keep=False: mx.min(x, axis=tuple(d) or None, keepdims=keep)
OPS[aten.any.dim] = lambda x, d, keep=False: mx.any(x, axis=d, keepdims=keep)
OPS[aten.all.dim] = lambda x, d, keep=False: mx.all(x, axis=d, keepdims=keep)
OPS[aten.argmax.default] = lambda x, d=None, keep=False: mx.argmax(x, axis=d, keepdims=keep)
OPS[aten.argmin.default] = lambda x, d=None, keep=False: mx.argmin(x, axis=d, keepdims=keep)
OPS[aten.var.correction] = lambda x, d=None, correction=1, keepdim=False: mx.var(
    x, axis=tuple(d) if d else None, keepdims=keepdim, ddof=int(correction or 0)
)


@_reg(aten.var_mean.correction)
def _var_mean(
    x: mx.array, d: Sequence[int] | None = None, correction: float = 1, keepdim: bool = False
) -> tuple[mx.array, mx.array]:
    """Variance and mean in float32 (instance norms over millions of voxels), cast back to the input dtype."""
    y = x.astype(mx.float32)
    axis = tuple(d) if d else None
    return (
        mx.var(y, axis=axis, keepdims=keepdim, ddof=int(correction or 0)).astype(x.dtype),
        mx.mean(y, axis=axis, keepdims=keepdim).astype(x.dtype),
    )


@_reg(aten._linalg_svd.default)
def _linalg_svd(
    a: mx.array, full_matrices: bool = False, compute_uv: bool = True, driver: str | None = None
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched SVD (CPU stream: tiny matrices such as rotation orthogonalization), float32."""
    del driver
    u, s, vh = mx.linalg.svd(a.astype(mx.float32), compute_uv=True, stream=mx.cpu)
    if not compute_uv:
        return mx.zeros((0,)), s.astype(a.dtype), mx.zeros((0,))
    del full_matrices
    return u.astype(a.dtype), s.astype(a.dtype), vh.astype(a.dtype)


@_reg(aten.unfold.default)
def _unfold(x: mx.array, dimension: int, size: int, step: int) -> mx.array:
    """`Tensor.unfold`: sliding windows of `size` along `dimension`, appended as the last axis."""
    dim = dimension % x.ndim if x.ndim else 0
    n = (x.shape[dim] - size) // step + 1
    idx = (mx.arange(n)[:, None] * step + mx.arange(size)[None]).reshape(-1)
    out = mx.take(x, idx, axis=dim)
    out = out.reshape(*x.shape[:dim], n, size, *x.shape[dim + 1 :])
    return mx.moveaxis(out, dim + 1, -1)


@_reg(aten.linalg_cholesky_ex.default)
def _cholesky_ex(a: mx.array, *, upper: bool = False, check_errors: bool = False) -> tuple[mx.array, mx.array]:
    """Cholesky factor (CPU stream, float32) and a zero `info`."""
    del check_errors
    lower = mx.linalg.cholesky(a.astype(mx.float32), upper=upper, stream=mx.cpu)
    return lower.astype(a.dtype), mx.zeros(a.shape[:-2], dtype=mx.int32)


@_reg(aten.linalg_cholesky.default)
def _cholesky(a: mx.array, *, upper: bool = False) -> mx.array:
    """Cholesky factor (CPU stream, float32)."""
    return mx.linalg.cholesky(a.astype(mx.float32), upper=upper, stream=mx.cpu).astype(a.dtype)


@_reg(aten.cholesky_solve.default)
def _cholesky_solve(b: mx.array, factor: mx.array, upper: bool = False) -> mx.array:
    """`A^-1 b` from the Cholesky factor of `A` by two triangular solves (CPU stream, float32)."""
    f, rhs = factor.astype(mx.float32), b.astype(mx.float32)
    lower = f.swapaxes(-1, -2) if upper else f
    y = mx.linalg.solve_triangular(lower, rhs, upper=False, stream=mx.cpu)
    x = mx.linalg.solve_triangular(lower.swapaxes(-1, -2), y, upper=True, stream=mx.cpu)
    return x.astype(b.dtype)


@_reg(aten._linalg_det.default)
def _linalg_det(a: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Determinant of batched 2x2 / 3x3 matrices by cofactor expansion (the LU outputs are unused placeholders)."""
    n = a.shape[-1]
    if n == 2:
        det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    elif n == 3:
        det = (
            a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
            - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
            + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
        )
    else:
        msg = f"determinant of {n}x{n} matrices"
        raise UnsupportedOp(msg)
    return det, mx.zeros((0,)), mx.zeros((0,))


OPS[aten.max.dim] = lambda x, d, keep=False: (mx.max(x, axis=d, keepdims=keep), mx.argmax(x, axis=d, keepdims=keep))
OPS[aten.max.default] = lambda x: mx.max(x)
OPS[aten.min.dim] = lambda x, d, keep=False: (mx.min(x, axis=d, keepdims=keep), mx.argmin(x, axis=d, keepdims=keep))
OPS[aten.any.default] = lambda x: mx.any(x)
OPS[aten.all.default] = lambda x: mx.all(x)
OPS[aten.empty_permuted.default] = lambda size, physical_layout, dtype=None, **_: mx.zeros(
    tuple(size), dtype=_dt(dtype) or mx.float32
)
OPS[aten.min.default] = lambda x: mx.min(x)


@_reg(aten.sort.default, aten.sort.stable)
def _sort(x: mx.array, *args: Any, **kw: Any) -> tuple[mx.array, mx.array]:
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
def _topk(x: mx.array, k: int, dim: int = -1, largest: bool = True, sorted_: bool = True) -> tuple[mx.array, mx.array]:
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


SDPA_FUSED_DIMS = (64, 72, 80, 96, 128)
"""Head sizes `mx.fast.scaled_dot_product_attention` runs fused (measured, MLX 0.32); others materialise the scores."""
SDPA_BLOCK_BYTES = 128 << 20
"""Float32 scores per query block of an unfused attention whose whole score matrix would exceed twice this."""


@_reg(aten.scaled_dot_product_attention.default)
def _sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
    dropout: float = 0.0,
    causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> mx.array:
    del dropout, enable_gqa
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    if causal:
        mask = "causal"
    elif mask is not None and mask.dtype not in {mx.bool_, q.dtype}:
        mask = mask.astype(q.dtype)
    d = q.shape[-1]
    if d in SDPA_FUSED_DIMS or q.ndim != 4 or isinstance(mask, str):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if d < SDPA_FUSED_DIMS[-1] and k.shape[-1] == d and v.shape[-1] == d:
        # zero-padded to the next fused head size: the same scores, no materialised score matrix (and faster)
        pad = [(0, 0)] * 3 + [(0, next(f for f in SDPA_FUSED_DIMS if f > d) - d)]
        out = mx.fast.scaled_dot_product_attention(
            mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad), scale=scale, mask=mask
        )
        return out[..., :d]
    b, h, lq, _ = q.shape
    row = b * h * k.shape[-2] * 4  # bytes of one query's float32 scores
    if lq * row <= 2 * SDPA_BLOCK_BYTES:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    # a head size MLX does not fuse materialises the scores: query blocks one after the other (`mx.depends`) bound them
    blk = max(1, SDPA_BLOCK_BYTES // row)
    per_query = isinstance(mask, mx.array) and mask.ndim >= 2 and mask.shape[-2] == lq
    outs: list[mx.array] = []
    for i in range(0, lq, blk):
        qi = q[:, :, i : i + blk]
        if outs:
            qi = mx.depends(qi, outs[-1])
        mi = mask[..., i : i + blk, :] if per_query else mask
        outs.append(mx.fast.scaled_dot_product_attention(qi, k, v, scale=scale, mask=mi))
    return mx.concatenate(outs, axis=2)


WIN_ATTN_OFF = bool(os.environ.get("FF_MLX_NO_WIN_ATTN"))  # masked short attention through `_sdpa` (summed mask)
# longest window the kernel runs: a row of scores per simdgroup spills its registers beyond (HAT's 16x16 windows,
# L = 256, D = 24, fp16: 212 against 46 ms for MLX sdpa; L = 64 / 96: 13 / 12 against 18 / 28 ms)
WIN_ATTN_MAX_L = int(os.environ.get("FF_MLX_WIN_ATTN_MAX_L", "96"))


def _win_attn_ok(q: mx.array, k: mx.array, v: mx.array, parts: tuple[mx.array, ...]) -> bool:
    """Whether `ops.window_attention` runs this attention with these mask parts (see `_sdpa_parts`)."""
    from fastforks.kernels.graph_mlx import ops

    b, h, lq, d = q.shape
    return (
        not WIN_ATTN_OFF
        and d not in SDPA_FUSED_DIMS
        and lq <= WIN_ATTN_MAX_L
        and len(parts) <= 2
        and all(
            p.ndim == 4 and b % p.shape[0] == 0 and p.shape[1] in (1, h) and tuple(p.shape[2:]) == (lq, lq)
            and p.dtype in (mx.float16, mx.bfloat16, mx.float32)
            for p in parts
        )
        and ops.window_attention_ok(q, k, v)
    )  # fmt: skip


def _sdpa_qkv(qkv: mx.array, *parts: mx.array, scale: float | None = None) -> mx.array:
    """`_sdpa_parts` of q, k and v sliced out of `qkv.permute(2, 0, 3, 1, 4)` (`[B, L, 3, H, D]`, `_sdpa_qkv_args`).

    The window kernel reads them where they lie; MLX would copy each slice of the permuted array (19 of 31 ms per
    SwinIR-L block at 512x512 fp16).
    """
    q, k, v = qkv.transpose(2, 0, 3, 1, 4)
    if _win_attn_ok(q, k, v, parts):
        from fastforks.kernels.graph_mlx import ops

        return ops.window_attention(qkv, None, None, scale if scale is not None else q.shape[3] ** -0.5, list(parts))
    return _sdpa_parts(q, k, v, *parts, scale=scale)


def _sdpa_parts(q: mx.array, k: mx.array, v: mx.array, *parts: mx.array, scale: float | None = None) -> mx.array:
    """`_sdpa` whose additive mask is `sum(parts)` broadcast and tiled over the batch (`_sdpa_mask_parts`).

    Windowed attention (Swin and its kin: `bias[None] + shift_mask[:, None]`, expanded to the batch) runs our
    kernel, which reads each part where it lies instead of a materialised `[B, H, L, L]` mask; it is used for the
    head sizes MLX does not fuse (it zero-pads them to 64, over 1.7x slower at `L = 64, D = 32`).
    """
    if _win_attn_ok(q, k, v, parts):
        from fastforks.kernels.graph_mlx import ops

        return ops.window_attention(q, k, v, scale if scale is not None else q.shape[3] ** -0.5, list(parts))
    b = q.shape[0]
    # in the run dtype, as `_sdpa` would cast the sum: an eager half-precision run never holds a float32 [B, H, L, L]
    mask = functools.reduce(operator.add, [p if p.dtype == q.dtype else p.astype(q.dtype) for p in parts])
    if mask.ndim == 4 and mask.shape[0] not in (1, b):
        mask = mx.tile(mask, (b // mask.shape[0], 1, 1, 1))
    return _sdpa(q, k, v, mask, scale=scale)


@_reg(aten._softmax.default, aten.softmax.int)
def _softmax(x: mx.array, dim: int, half_to_float: bool = False) -> mx.array:
    del half_to_float
    return mx.softmax(x, axis=dim, precise=True)


OPS[aten._log_softmax.default] = lambda x, d, h=False: (
    x - mx.logsumexp(x.astype(mx.float32), axis=d, keepdims=True).astype(x.dtype)
)


# ---- normalisation ----------------------------------------------------------------------------------------------
ROW_LN = not os.environ.get("FF_MLX_NO_ROW_LN")  # narrow layer norms through `mx.fast.layer_norm`


def _ops() -> Any:
    from fastforks.kernels.graph_mlx import ops

    return ops


@_reg(aten.native_layer_norm.default)
def _layer_norm(
    x: mx.array, shape: Sequence[int], w: mx.array | None, b: mx.array | None, eps: float
) -> tuple[mx.array, None, None]:
    if len(shape) == 1:
        if (
            ROW_LN
            and x.shape[-1] <= _ops().ROW_LN_MAX
            and x.dtype in (mx.float16, mx.bfloat16, mx.float32)
            and all(t is None or t.dtype == x.dtype for t in (w, b))
        ):
            return _ops().row_layer_norm(x, w, b, eps), None, None
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
    # fast layer norm over each group's row: float32 statistics inside the kernel, no float32 copy of the activation
    y = mx.fast.layer_norm(x.reshape(shape[0] * groups, -1), None, None, eps).reshape(shape)
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
    *rest: Any,
) -> tuple[mx.array, None, None]:
    eps = rest[-1]
    shape = [1, -1] + [1] * (x.ndim - 2)
    scale = mx.rsqrt(var.astype(mx.float32) + eps)
    if w is not None:
        scale = scale * w.astype(mx.float32)
    shift = -mean.astype(mx.float32) * scale + (b.astype(mx.float32) if b is not None else 0)
    return x * scale.astype(x.dtype).reshape(shape) + shift.astype(x.dtype).reshape(shape), None, None


STATS_CHUNK_MB = 64  # float32 rows per partial sum of `_batch_norm_stats_lean`


@functools.partial(mx.compile, shapeless=True)
def _affine_cast(x: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    return (x.astype(mx.float32) * scale + shift).astype(x.dtype)


def _batch_norm_stats_lean(x: mx.array, w: mx.array | None, b: mx.array | None, eps: float) -> mx.array:
    """`_batch_norm_batch_stats` without float32 copies of the map.

    The float32 statistics are partial sums over row blocks; the normalisation and the affine are one fused pass.
    The plain form held three float32 copies (C-FoundationStereo's 534 MB fp16 volume: peak 3.7 GB -> 0.6 GB).
    """
    c = x.shape[1]
    flat = mx.moveaxis(x, 1, -1).reshape(-1, c)
    rows = flat.shape[0]
    step = max(1, (STATS_CHUNK_MB << 20) // (4 * c))
    blocks = [flat[i : i + step] for i in range(0, rows, step)]
    mean = sum(mx.sum(f.astype(mx.float32), axis=0) for f in blocks) / rows
    var = sum(mx.sum(mx.square(f.astype(mx.float32) - mean), axis=0) for f in blocks) / rows
    scale = mx.rsqrt(mx.maximum(var, 0) + eps)
    if w is not None:
        scale = scale * w.astype(mx.float32)
    shift = -mean * scale
    if b is not None:
        shift = shift + b.astype(mx.float32)
    return mx.moveaxis(_affine_cast(flat, scale, shift).reshape(*x.shape[:1], *x.shape[2:], c), -1, 1)


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
    shape = [1, -1] + [1] * (x.ndim - 2)
    c = x.shape[1]
    xl = mx.moveaxis(x, 1, -1)  # the statistics over the channels-last view: the layout convolutions leave behind
    flat = xl.reshape(-1, c)
    if not (_KERNELS and flat.shape[0] >= 4096):
        return _batch_norm_stats_lean(x, w, b, eps), None, None
    from fastforks.kernels.graph_mlx import ops

    mean, var = ops.channel_stats(flat)
    y = ((xl.astype(mx.float32) - mean) * mx.rsqrt(var + eps)).astype(x.dtype)
    y = mx.moveaxis(y, -1, 1)
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


WINO_TILE = int(os.environ.get("FF_WINO_M", "4"))  # Winograd output tile (2 or 4) of the kernels-mode 3x3 convolution
_WINO: dict[
    Any, tuple[mx.array, mx.array]
] = {}  # (id(weight), tile) -> (weight, its Winograd-domain form), per GraphNet
WINO_LAZY_MB = float(os.environ.get("FF_WINO_LAZY_MB", "4"))  # transformed weights above this size are made per call
WINO_SMALL_HW = 16 * 16  # a map of at most this many pixels uses F(2,3): its weights are 1.8x (not 4x) the 3x3 kernel


def _wino_ok(
    xl: mx.array, w: mx.array, stride: Sequence[int], padding: Sequence[int], dilation: Sequence[int], groups: int
) -> bool:
    """Whether the kernels-mode Winograd path takes this (NHWC input) convolution: dense 3x3, stride 1, pad 1."""
    if groups != 1 or tuple(w.shape[2:]) != (3, 3) or tuple(stride) != (1, 1) or tuple(padding) != (1, 1):
        return False
    if tuple(dilation) != (1, 1) or xl.shape[3] != w.shape[1] or w.shape[0] > 1024 or w.shape[1] > 1024:
        return False
    from fastforks.kernels.conv_mlx import ops as conv_ops

    return conv_ops.supported(tuple(xl.shape), w.shape[0])


def _direct_ok(cin: int, hw: int, wshape: Sequence[int], stride: Sequence[int]) -> bool:
    """Whether the kernels-mode direct few-channel conv takes this dense 2-D conv (`hw` input pixels, OIHW `wshape`)."""
    if len(wshape) != 4 or cin != wshape[1] or hw < DIRECT_MIN_HW or DIRECT_OFF:
        return False
    from fastforks.kernels.conv_mlx import ops as conv_ops

    return conv_ops.direct_supported(wshape[1], wshape[0], tuple(wshape[2:]), tuple(stride))


DIRECT_MIN_HW = 128 * 128  # smaller maps: launch-bound either way, `mx.conv2d` keeps them
DIRECT_OFF = bool(os.environ.get("FF_MLX_NO_DIRECT"))  # kernels mode without the direct few-channel conv


def _wino3d_ok(
    xl: mx.array, w: mx.array, stride: Sequence[int], padding: Sequence[int], dilation: Sequence[int], groups: int
) -> bool:
    """Whether the kernels-mode 3x3x3 Winograd path takes this (NDHWC input) convolution: dense, stride 1, pad 1."""
    if not _KERNELS or groups != 1 or tuple(w.shape[2:]) != (3, 3, 3) or tuple(stride) != (1, 1, 1):
        return False
    if tuple(padding) != (1, 1, 1) or tuple(dilation) != (1, 1, 1) or xl.shape[4] != w.shape[1]:
        return False
    from fastforks.kernels.conv_mlx import ops as conv_ops

    return conv_ops.supported3d(tuple(xl.shape), w.shape[0], xl.dtype.size)


def _wino3d_weights(w: mx.array, dtype: mx.Dtype) -> mx.array:
    """The stacked-tap Winograd weights of the constant 3x3x3 kernel `w` (cached like `_wino_weights`)."""
    hit = _WINO.get((-id(w), 4))
    if hit is None:
        from fastforks.kernels.conv_mlx import ops as conv_ops

        hit = _WINO[-id(w), 4] = (w, conv_ops.weights3d(mx.transpose(w, (0, 2, 3, 4, 1)), 4, dtype))
    return hit[1]


def _wino_tile(xl: mx.array) -> int:
    """The Winograd tile of this NHWC input.

    F(2,3) on tiny maps (16x16 with 512 channels: 0.20 ms against 0.36 for F(4,3), and 8 MB of weights per layer,
    not 18), else `WINO_TILE`.
    """
    return 2 if xl.shape[1] * xl.shape[2] <= WINO_SMALL_HW else WINO_TILE


def _wino_weights(w: mx.array, wl: mx.array, dtype: mx.Dtype, m: int) -> mx.array:
    """The tile-`m` Winograd-domain weights of constant `w`, made on first use (also while tracing a segment)."""
    hit = _WINO.get((id(w), m))
    if hit is None:
        from fastforks.kernels.conv_mlx import ops as conv_ops

        if (m + 2) ** 2 * w.shape[0] * w.shape[1] * dtype.size > WINO_LAZY_MB * (1 << 20):
            # a big layer's transformed weights are not held (4x the kernel for F(4,3)): one layer's transient per call
            return conv_ops.weights_graph(wl, m, dtype)
        hit = _WINO[id(w), m] = (w, conv_ops.weights(wl, m, dtype))  # `w` stays referenced so its id is not reused
    return hit[1]


KEEP_F32_CONV = bool(os.environ.get("FF_MLX_KEEP_F32_CONV"))  # float32 inputs of half-precision convolutions stay


def _conv_input(x: mx.array, w: mx.array) -> mx.array:
    """A float32 input of a half-precision convolution in the weights' dtype, as PyTorch's half-precision run has it.

    The replayed twin is float32: a `.to(dtype)` before a convolution, a float32 state buffer (`torch.empty(...,
    dtype=dt)`) or a float32 lookup result is recorded as float32, and the float16 replay then promoted every layer it
    feeds, the whole recurrent update block of the RAFT / IGEV / MonSter families (79-92 % of the convolution FLOPs in
    float32 at 384x576). PyTorch cannot convolve float32 with float16 weights, so its float16 run casts there too.
    """
    if not KEEP_F32_CONV and x.dtype == mx.float32 and w.dtype in (mx.float16, mx.bfloat16):
        return x.astype(w.dtype)
    return x


QKV_OFF = bool(os.environ.get("FF_MLX_NO_WIN_QKV"))  # window attention on sliced (copied) q, k, v
WT_OFF = bool(os.environ.get("FF_MLX_NO_WT"))  # linear weights stay `[out, in]`, read through a transposed view


def _transposed_weights(nodes: list[Any], names: dict[str, Any]) -> dict[str, str]:
    """`permute(w, [1, 0])` / `t(w)` nodes of 2-D weights that are read only that way: node name -> weight target.

    Every `nn.Linear` exports as `mm` / `addmm` of the transposed weight. MLX's GEMM reads a transposed `[out, in]`
    operand 4-5 % slower in float32 (3 % in float16) than a contiguous `[in, out]` one (M1 Pro, ViT shapes), so the
    translator stores these weights transposed once, in place of the original.
    """
    if WT_OFF:
        return {}
    out: dict[str, str] = {}
    for p in nodes:
        tgt = names.get(p.name) if p.op == "placeholder" else None
        val = p.meta.get("val") if tgt is not None else None
        if not isinstance(val, torch.Tensor) or val.ndim != 2 or not val.is_floating_point() or not p.users:
            continue
        users = list(p.users)
        if all(
            u.target is aten.t.default or (u.target is aten.permute.default and list(u.args[1]) == [1, 0])
            for u in users
        ):
            out.update((u.name, str(tgt)) for u in users)
    return out


def _sdpa_mask_parts(nodes: list[Any]) -> tuple[dict[str, tuple[Callable[..., Any], list[Any]]], set[str]]:
    """Attention nodes whose float mask is `expand(repeat(a + b))` (each step optional, sole users): their call.

    `_sdpa_parts` runs them on the parts, so the peeled nodes (returned too) never build the `[B, H, L, L]` mask;
    `_sdpa_qkv` when q, k and v are slices of one fused projection (`_sdpa_qkv_args`), which is peeled as well.
    """
    out: dict[str, tuple[Callable[..., Any], list[Any]]] = {}
    skip: set[str] = set()
    if WIN_ATTN_OFF:
        return out, skip

    def call(n: Any, target: Any) -> bool:
        return isinstance(n, torch.fx.Node) and n.target is target and len(n.users) == 1

    for node in nodes:
        a = node.args
        if (
            node.op != "call_function"
            or node.target is not aten.scaled_dot_product_attention.default
            or len(a) != 4
            or set(node.kwargs) - {"scale"}
        ):
            continue
        m, peeled = a[3], []
        if call(m, aten.expand.default):
            peeled.append(m)
            m = m.args[0]
        if call(m, aten.repeat.default) and list(m.args[1])[1:] == [1, 1, 1]:
            peeled.append(m)
            m = m.args[0]
        terms = [m]
        if call(m, aten.add.Tensor) and not m.kwargs and all(isinstance(t, torch.fx.Node) for t in m.args):
            peeled.append(m)
            terms = list(m.args)
        vals = [t.meta.get("val") if isinstance(t, torch.fx.Node) else None for t in terms]
        if not peeled or not all(isinstance(v, torch.Tensor) and v.ndim == 4 and v.is_floating_point() for v in vals):
            continue
        qkv = None if QKV_OFF else _sdpa_qkv_args(node)
        if qkv is not None:
            out[node.name] = (_sdpa_qkv, [qkv[0], *terms])
            peeled += qkv[1]
        else:
            out[node.name] = (_sdpa_parts, [*a[:3], *terms])
        skip.update(p.name for p in peeled)
    return out, skip


def _sdpa_qkv_args(node: Any) -> tuple[Any, list[Any]] | None:
    """`(qkv, peeled)` when the attention's q, k, v are `qkv.permute(2, 0, 3, 1, 4)[0 / 1 / 2]` (sole users).

    The fused qkv projection of Swin and ViT blocks; the slice is `select`, or `slice` + `squeeze`.
    """
    peeled: list[Any] = []
    perm = None
    for i, arg in enumerate(node.args[:3]):
        chain, t = [], arg
        if isinstance(t, torch.fx.Node) and t.target is aten.squeeze.dims and list(t.args[1]) == [0]:
            chain.append(t)
            t = t.args[0]
            ok = t.target is aten.slice.Tensor and list(t.args[1:]) in ([0, i, i + 1], [0, i, i + 1, 1])
        else:
            ok = isinstance(t, torch.fx.Node) and t.target is aten.select.int and list(t.args[1:]) == [0, i]
        if not ok or len(t.users) != 1:
            return None
        chain.append(t)
        p = t.args[0]
        if perm is None:
            perm = p
        if p is not perm:
            return None
        peeled += chain
    if not (
        perm.target is aten.permute.default and list(perm.args[1]) == [2, 0, 3, 1, 4] and len(perm.users) == 3
        and all(u in peeled for u in perm.users)
    ):  # fmt: skip
        return None
    base = perm.args[0]
    val = base.meta.get("val") if isinstance(base, torch.fx.Node) else None
    if not isinstance(val, torch.Tensor) or val.ndim != 5 or val.shape[2] != 3:
        return None
    return base, [*peeled, perm]


def _conv_only_cats(nodes: list[Any]) -> set[str]:
    """`aten.cat` nodes whose value only feeds convolution inputs (directly or through other such cats).

    The convolution casts a float32 input to its half-precision weights (`_conv_input`); a cast commutes with the
    concatenation, so these cats can cast their float32 operands first and move half the bytes. IGEV++'s update block
    concatenated the float32 disparity into its 128- and 384-channel GRU inputs: 1.0 s of float32 cats per call at
    994x1482.
    """
    out: set[str] = set()
    for node in reversed(nodes):
        if node.op != "call_function" or node.target is not aten.cat.default or not node.users:
            continue
        if all(
            u.name in out or (u.target is aten.convolution.default and u.args[0] is node and node not in u.args[1:])
            for u in node.users
        ):
            out.add(node.name)
    return out


def _cat_in(op: Callable[..., mx.array], dtype: mx.Dtype, xs: Sequence[mx.array], *a: Any, **kw: Any) -> mx.array:
    """`op` (a cat) with its float32 operands cast to the half-precision `dtype` first (see `_conv_only_cats`)."""
    return op([x.astype(dtype) if x.dtype == mx.float32 else x for x in xs], *a, **kw)


@_reg(aten.convolution.default)
def _conv(
    x: mx.array,
    w: mx.array,
    b: mx.array | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> mx.array:
    return _conv_act(x, w, b, stride, padding, dilation, transposed, output_padding, groups)


def _conv_act(
    x: mx.array,
    w: mx.array,
    b: mx.array | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    act: int = 0,
    slope: mx.array | None = None,
) -> mx.array:
    """`_conv` followed by `act` (0 none, 1 ReLU, 2 PReLU with the per-channel `slope`; kernels mode folds it in)."""
    if act:
        y = _conv_core(x, w, b, stride, padding, dilation, transposed, output_padding, groups, act, slope)
        if y is not None:
            return y
        y = _conv(x, w, b, stride, padding, dilation, transposed, output_padding, groups)
        if act == 1:
            return mx.maximum(y, 0)
        return mx.where(y > 0, y, y * slope.reshape(1, -1, 1, 1))  # ty: ignore[unresolved-attribute]
    return _conv_core(x, w, b, stride, padding, dilation, transposed, output_padding, groups)  # ty: ignore[invalid-return-type]


def _conv_res(
    x: mx.array,
    w: mx.array,
    b: mx.array | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    residual: mx.array,
) -> mx.array:
    """`_conv` plus `residual`; kernels mode adds it in the Winograd output transform."""
    x = _conv_input(x, w)
    if _KERNELS:
        xl = _nhwc(x)
        if _wino_ok(xl, w, stride, padding, dilation, groups):
            from fastforks.kernels.conv_mlx import ops as conv_ops

            wl = mx.transpose(w, (0, 2, 3, 1))
            m = _wino_tile(xl)
            return _nchw(conv_ops.conv3x3(xl, _wino_weights(w, wl, xl.dtype, m), b, m=m, residual=_nhwc(residual)))
    return _conv(x, w, b, stride, padding, dilation, transposed, output_padding, groups) + residual


def _channel_slice(y: mx.array, lo: int, hi: int) -> mx.array:
    """Output channels `lo:hi` of a merged convolution (a view)."""
    return y[:, lo:hi]


def _shift_pad(x: mx.array, axis: int, lo: int, hi: int) -> mx.array:
    """Zero-pad `axis` by `lo` / `hi` (a negative amount crops instead)."""
    if lo < 0 or hi < 0:
        n = x.shape[axis]
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(max(-lo, 0), n - max(-hi, 0))
        x = x[tuple(idx)]
    if lo > 0 or hi > 0:
        widths = [(0, 0)] * x.ndim
        widths[axis] = (max(lo, 0), max(hi, 0))
        x = mx.pad(x, widths)
    return x


def _depthwise3d(x: mx.array, w: mx.array, stride: Sequence[int], padding: Sequence[int]) -> mx.array:
    """NDHWC depthwise conv3d (`w` `[C, 1, kd, kh, kw]`): a sum over the taps of strided slices times each tap.

    The per-group `mx.conv3d` loop copies a channel slice of the whole volume per group (FoundationStereo's `conv_patch`
    k4 s4 on a 104x256x376x28 volume: 3.4 GB of transients, 440 ms; this: one output-sized accumulator, 36 ms).
    """
    if any(padding):
        x = mx.pad(x, [(0, 0), *((p, p) for p in padding), (0, 0)])
    k = w.shape[2:]
    out_dims = [(n - kk) // s + 1 for n, kk, s in zip(x.shape[1:4], k, stride, strict=True)]
    wl = mx.transpose(w[:, 0], (1, 2, 3, 0))  # [kd, kh, kw, C]
    acc = None
    for a in range(k[0]):
        for b in range(k[1]):
            for c in range(k[2]):
                sl = x[
                    :,
                    a : a + (out_dims[0] - 1) * stride[0] + 1 : stride[0],
                    b : b + (out_dims[1] - 1) * stride[1] + 1 : stride[1],
                    c : c + (out_dims[2] - 1) * stride[2] + 1 : stride[2],
                ]
                t = sl * wl[a, b, c]
                acc = t if acc is None else acc + t
    return acc  # ty: ignore[invalid-return-type]


TRANSPOSE3D_SLAB_MB = 96


def _conv_transpose3d_phases(
    x: mx.array, w: mx.array, stride: Sequence[int], padding: Sequence[int], output_padding: Sequence[int]
) -> mx.array | None:
    """NDHWC transposed conv as one small stride-1 conv per output phase (`mx.conv_transpose3d` is ~8x slower).

    `w` is `[Cin, Cout, kd, kh, kw]`; returns None when the output is not `stride` x the input (no phase grid).
    A big output is made in depth slabs (input rows with a halo, zero beyond the ends as the full conv sees them):
    the phases, their stack and interleave of one slab are its only transients (FoundationStereo's `conv1_up` at
    104x256x376x28: 3.1 GB of transients whole, the slabs' concatenation ~0.6 GB).
    """
    dims = x.shape[1:4]
    k = w.shape[2:]
    if any(
        (n - 1) * s - 2 * p + kk + op != n * s
        for n, s, p, kk, op in zip(dims, stride, padding, k, output_padding, strict=False)
    ):
        return None
    d = dims[0]
    out_bytes = x.shape[0] * d * stride[0] * dims[1] * stride[1] * dims[2] * stride[2] * w.shape[1] * x.dtype.size
    rows = max(1, int(d * (TRANSPOSE3D_SLAB_MB << 20) // max(out_bytes, 1)))
    if rows < d:
        halo = (k[0] - 1) // stride[0] + 1
        slabs = []
        for d0 in range(0, d, rows):
            d1 = min(d, d0 + rows)
            a, b = d0 - halo, d1 + halo
            xs = x[:, max(a, 0) : min(b, d)]
            if a < 0 or b > d:
                xs = mx.pad(xs, [(0, 0), (max(-a, 0), max(b - d, 0)), (0, 0), (0, 0), (0, 0)])
            y = _conv_transpose3d_phases_whole(xs, w, stride, padding)
            slabs.append(y[:, halo * stride[0] : (halo + d1 - d0) * stride[0]])
        return mx.concatenate(slabs, axis=1)
    return _conv_transpose3d_phases_whole(x, w, stride, padding)


def _conv_transpose3d_phases_whole(x: mx.array, w: mx.array, stride: Sequence[int], padding: Sequence[int]) -> mx.array:
    """`_conv_transpose3d_phases` of the whole input in one go."""
    k = w.shape[2:]
    wt = mx.transpose(w, (1, 2, 3, 4, 0))  # [Cout, kd, kh, kw, Cin]
    taps = []  # per dim, per phase: (kernel indices ordered by input offset, lo pad, hi pad)
    for ax in range(3):
        s, p, kk = stride[ax], padding[ax], k[ax]
        per = []
        for r in range(s):
            ks = sorted((j for j in range(kk) if (r + p - j) % s == 0), reverse=True)  # increasing input offset
            per.append((ks, -((r + p - ks[0]) // s), (r + p - ks[-1]) // s) if ks else None)
        taps.append(per)
    out = []
    for rd in range(stride[0]):
        for rh in range(stride[1]):
            for rw in range(stride[2]):
                t = (taps[0][rd], taps[1][rh], taps[2][rw])
                if any(e is None for e in t):
                    out.append(mx.zeros((*x.shape[:4], wt.shape[0]), x.dtype))
                    continue
                sub = wt[:, t[0][0]][:, :, t[1][0]][:, :, :, t[2][0]]
                xp = x
                for ax in range(3):
                    xp = _shift_pad(xp, 1 + ax, t[ax][1], t[ax][2])
                out.append(mx.conv3d(xp, sub))
    sd, sh, sw = stride
    n, d, h, wd = x.shape[:4]
    y = mx.stack(out).reshape(sd, sh, sw, n, d, h, wd, -1)
    return mx.transpose(y, (3, 4, 0, 5, 1, 6, 2, 7)).reshape(n, d * sd, h * sh, wd * sw, -1)


def _conv_transpose2d_phases(
    x: mx.array, w: mx.array, stride: Sequence[int], padding: Sequence[int], output_padding: Sequence[int]
) -> mx.array | None:
    """NHWC transposed conv of ONE input channel (`w` `[1, Cout, kh, kw]`) as shifted multiply-adds per output phase.

    None for more input channels or when the output is not `stride` x the input (no phase grid). stride == kernel
    without padding is a broadcast multiply and a depth-to-space. TEED's single-channel up-sampling, 640 -> 1280
    float16: k 2 2.7 -> 0.6 ms, k 4 6.5 -> 1.0 ms (`mx.conv_transpose2d`); with 16 channels MLX's own is faster.
    """
    n, h, wd, cin = x.shape
    k = w.shape[2:]
    dims = (h, wd)
    if cin != 1 or any(
        (m - 1) * s - 2 * p + kk + op != m * s
        for m, s, p, kk, op in zip(dims, stride, padding, k, output_padding, strict=True)
    ):
        return None
    cout = w.shape[1]
    sh, sw = stride
    if tuple(stride) == tuple(k) and not any(padding):
        wm = mx.transpose(w, (0, 2, 3, 1))  # [Cin, kh, kw, Cout]
        y = x.reshape(n, h, wd, 1, 1, 1) * wm[0]
        y = mx.transpose(y.reshape(n, h, wd, sh, sw, cout), (0, 1, 3, 2, 4, 5))
        return y.reshape(n, h * sh, wd * sw, cout)
    wt = mx.transpose(w, (1, 2, 3, 0))  # [Cout, kh, kw, 1]
    taps = []  # per dim, per phase: (kernel indices ordered by input offset, lo pad, hi pad)
    for ax in range(2):
        s, p, kk = stride[ax], padding[ax], k[ax]
        per = []
        for r in range(s):
            ks = sorted((j for j in range(kk) if (r + p - j) % s == 0), reverse=True)  # increasing input offset
            per.append((ks, -((r + p - ks[0]) // s), (r + p - ks[-1]) // s) if ks else None)
        taps.append(per)
    out = []
    for rh in range(sh):
        for rw in range(sw):
            t = (taps[0][rh], taps[1][rw])
            if t[0] is None or t[1] is None:
                out.append(mx.zeros((n, h, wd, cout), x.dtype))
                continue
            sub = wt[:, t[0][0]][:, :, t[1][0]]  # [Cout, th, tw, Cin]
            xp = _shift_pad(_shift_pad(x, 1, t[0][1], t[0][2]), 2, t[1][1], t[1][2])
            y = None
            for i in range(sub.shape[1]):
                for j in range(sub.shape[2]):
                    term = xp[:, i : i + h, j : j + wd] * sub[:, i, j, 0]
                    y = term if y is None else y + term
            out.append(y)
    y = mx.stack(out).reshape(sh, sw, n, h, wd, cout)
    return mx.transpose(y, (2, 3, 0, 4, 1, 5)).reshape(n, h * sh, wd * sw, cout)


def _conv_core(
    x: mx.array,
    w: mx.array,
    b: mx.array | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    act: int = 0,
    slope: mx.array | None = None,
) -> mx.array | None:
    """The convolution; with `act`, `None` unless the Winograd kernel ran (and applied the activation itself)."""
    x = _conv_input(x, w)
    if act and (
        x.ndim == 3
        or transposed
        or (x.ndim == 5 and not _wino3d_ok(mx.transpose(x, (0, 2, 3, 4, 1)), w, stride, padding, dilation, groups))
    ):
        return None
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
        elif groups > 1 and groups == xl.shape[-1] and stride[0] == 1 and w.shape[2] <= 7:
            # depthwise (channel multiplier w.shape[0] / groups): shifted multiply-adds that fuse with their neighbours
            # (TAPIR's PIPs mixer [64, 24, 2048], k = 3, fp32: 1.9 -> 0.8 ms; MLX's grouped conv1d is slow)
            k, dil, pad = w.shape[2], dilation[0], padding[0]
            if w.shape[0] != groups:
                xl = mx.repeat(xl, w.shape[0] // groups, axis=-1)
            xp = mx.pad(xl, [(0, 0), (pad, pad), (0, 0)]) if pad else xl
            out = xp.shape[1] - dil * (k - 1)
            y = None
            for t in range(k):
                term = xp[:, t * dil : t * dil + out] * w[:, 0, t]
                y = term if y is None else y + term
        else:
            y = mx.conv1d(xl, wl, stride=stride[0], padding=padding[0], dilation=dilation[0], groups=groups)
        if b is not None:
            y = y + b
        return mx.transpose(y, (0, 2, 1))
    if x.ndim == 5:  # 3-D: [N, C, D, H, W] -> NDHWC
        if (
            transposed
            and groups == 1
            and tuple(stride) == tuple(w.shape[2:])
            and not any(padding)
            and not any(output_padding)
            and all(d == 1 for d in dilation)
        ):
            # stride == kernel (nnU-Net's up-sampling): one matmul and a depth-to-space reshape
            # (`mx.conv_transpose3d` is wrong on large maps)
            n, _, d, h, wd = x.shape
            cin, cout, kd, kh, kw = w.shape
            xl = mx.transpose(x, (0, 2, 3, 4, 1)).reshape(-1, cin)
            wm = mx.transpose(w, (0, 2, 3, 4, 1)).reshape(cin, -1)  # columns (kd, kh, kw, cout)
            y = (xl @ wm).reshape(n, d, h, wd, kd, kh, kw, cout)
            y = mx.transpose(y, (0, 1, 4, 2, 5, 3, 6, 7)).reshape(n, d * kd, h * kh, wd * kw, cout)
        elif transposed and (
            groups == 1
            and all(d == 1 for d in dilation)
            and (y := _conv_transpose3d_phases(mx.transpose(x, (0, 2, 3, 4, 1)), w, stride, padding, output_padding))
            is not None
        ):
            pass
        elif transposed:
            y = mx.conv_transpose3d(
                mx.transpose(x, (0, 2, 3, 4, 1)),
                mx.transpose(w, (1, 2, 3, 4, 0)),  # [Cin, Cout/g, kd, kh, kw] -> [Cout/g, kd, kh, kw, Cin]
                stride=tuple(stride),
                padding=tuple(padding),
                dilation=tuple(dilation),
                output_padding=tuple(output_padding),
                groups=groups,
            )
        else:
            xl3 = mx.transpose(x, (0, 2, 3, 4, 1))
            wl3 = mx.transpose(w, (0, 2, 3, 4, 1))
            kw3 = {"stride": tuple(stride), "padding": tuple(padding), "dilation": tuple(dilation)}
            if groups == 1 and xl3.shape[-1] <= 2 and tuple(stride) == (1, 1, 1) and all(d == 1 for d in dilation):
                # few input channels (a CT / MRI stem): the depth taps folded into channels, one conv2d over the slices
                # (1 -> 32 channels, 224^3 float16: 130 -> 26 ms; `mx.conv3d` is slow at Cin 1-2)
                n, dd, hh, ww, cin = xl3.shape
                kd, pd = w.shape[2], padding[0]
                xp = mx.pad(xl3, [(0, 0), (pd, pd), (0, 0), (0, 0), (0, 0)]) if pd else xl3
                do = dd + 2 * pd - kd + 1
                xs = mx.concatenate([xp[:, a : a + do] for a in range(kd)], axis=-1).reshape(n * do, hh, ww, kd * cin)
                w2 = mx.transpose(w, (0, 3, 4, 2, 1)).reshape(w.shape[0], w.shape[3], w.shape[4], kd * cin)
                y = mx.conv2d(xs, w2, padding=tuple(padding[1:]))
                y = y.reshape(n, do, *y.shape[1:])
            elif _wino3d_ok(xl3, w, stride, padding, dilation, groups):
                from fastforks.kernels.conv_mlx import ops as conv_ops

                y = conv_ops.conv3d(xl3, _wino3d_weights(w, xl3.dtype), b, act=act, slope=slope)
                return mx.transpose(y, (0, 4, 1, 2, 3))
            elif groups == 1:
                y = mx.conv3d(xl3, wl3, **kw3)
            elif groups == xl3.shape[-1] == w.shape[0] and all(d == 1 for d in dilation):
                y = _depthwise3d(xl3, w, stride, padding)
            else:  # `mx.conv3d` has no groups: one conv per group
                cg, og = xl3.shape[-1] // groups, wl3.shape[0] // groups
                y = mx.concatenate(
                    [
                        mx.conv3d(xl3[..., g * cg : (g + 1) * cg], wl3[g * og : (g + 1) * og], **kw3)
                        for g in range(groups)
                    ],
                    axis=-1,
                )
        if b is not None:
            y = y + b
        return mx.transpose(y, (0, 4, 1, 2, 3))
    xl = _nhwc(x)
    y = None
    if transposed and groups == 1 and all(d == 1 for d in dilation):
        y = _conv_transpose2d_phases(xl, w, stride, padding, output_padding)
    if y is not None:
        pass
    elif transposed:
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
            and (groups < xl.shape[3] or xl.shape[3] <= 4)
            and w.shape[0] % groups == 0
            and _direct_ok(xl.shape[3], xl.shape[1] * xl.shape[2], (w.shape[0], xl.shape[3], *w.shape[2:]), stride)
        ):
            # few input channels in groups (TEED's 3 -> 24, groups 3): one dense direct conv with block-diagonal
            # weights beats the repeat + depthwise below (1280^2 float16 2.6 -> 1.1 ms)
            cg, og = xl.shape[3] // groups, w.shape[0] // groups
            mask = (mx.arange(w.shape[0])[:, None] // og) == (mx.arange(xl.shape[3])[None, :] // cg)
            wd = mx.where(mask[:, None, None, :], mx.tile(wl, (1, 1, 1, groups)), 0).astype(wl.dtype)
            from fastforks.kernels.conv_mlx import ops as conv_ops

            return _nchw(conv_ops.direct(xl, wd, b, tuple(stride), tuple(padding), tuple(dilation)))
        if 1 < groups == xl.shape[3] < w.shape[0] and w.shape[0] % groups == 0:
            # depthwise with a channel multiplier: repeat the input, then a plain depthwise conv (MLX's grouped conv
            # is slow there: TEED's 3 -> 24 at 1280^2 8.2 -> 3.8 ms, 48 -> 96 at 320^2 8.1 -> 1.3 ms, float16)
            xl = mx.repeat(xl, w.shape[0] // groups, axis=-1)
            groups = w.shape[0]
        if (
            (_KERNELS or DW_KERNEL)
            and not act
            and groups > 1
            and groups == xl.shape[3] == w.shape[0]
            and groups % 4 == 0
            and w.shape[2] * w.shape[3] > 1
        ):
            from fastforks.kernels.graph_mlx import ops

            bias = b if b is not None else mx.zeros((groups,), dtype=xl.dtype)
            return _nchw(
                ops.depthwise(
                    xl, mx.transpose(w[:, 0], (1, 2, 0)), bias, tuple(stride), tuple(padding), tuple(dilation)
                )
            )
        if _KERNELS and groups == 1 and _direct_ok(xl.shape[3], xl.shape[1] * xl.shape[2], w.shape, stride):
            from fastforks.kernels.conv_mlx import ops as conv_ops

            return _nchw(
                conv_ops.direct(xl, wl, b, tuple(stride), tuple(padding), tuple(dilation), act=act, slope=slope)
            )
        if _KERNELS and _wino_ok(xl, w, stride, padding, dilation, groups):
            from fastforks.kernels.conv_mlx import ops as conv_ops

            m = _wino_tile(xl)
            return _nchw(conv_ops.conv3x3(xl, _wino_weights(w, wl, xl.dtype, m), b, m=m, act=act, slope=slope))
        if act:
            return None
        y = _conv2d(xl, wl, tuple(stride), tuple(padding), tuple(dilation), groups)
    if b is not None:
        y = y + b
    return _nchw(y)


MERGE_OFF = bool(os.environ.get("FF_MLX_NO_MERGE"))  # sibling convolutions (same input and geometry) stay separate
ALIGN_OFF = bool(os.environ.get("FF_MLX_NO_ALIGN"))  # `mx.conv2d` with ragged channel counts as they are
WIDE_OUT_OFF = bool(os.environ.get("FF_MLX_NO_WIDE_OUT"))  # 32 / 48-output 3x3 convs on MLX's narrow tile
SPLIT_IN_OFF = bool(os.environ.get("FF_MLX_NO_SPLIT_IN"))  # very wide-input 3x3 convs as one `mx.conv2d`


def _mlx_winograd(
    wshape: Sequence[int],
    cout: int,
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    positions: int,
) -> bool:
    """Whether `mx.conv2d` runs an OHWI kernel of this shape with `cout` outputs as Winograd.

    MLX 0.32's dispatch: dense 3x3, stride and dilation 1, channels multiples of 32 summing to at least 256, at least
    4096 positions (checked against the float16 error, which jumps 17x exactly there).
    """
    cin = wshape[3]
    return (
        groups == 1
        and tuple(wshape[1:3]) == (3, 3)
        and stride == (1, 1)
        and dilation == (1, 1)
        and cin % 32 == 0
        and cout % 32 == 0
        and cin + cout >= 256
        and positions >= 4096
    )


def _conv2d(
    xl: mx.array,
    wl: mx.array,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> mx.array:
    """`mx.conv2d` with ragged channel counts padded to the widths its GEMM tiles take (zero weights, output sliced).

    MLX's implicit GEMM is much slower on output channels that are not a multiple of 16 and on wide inputs that are not
    a multiple of 32 (M1 Pro, float16, 256x376: 128 -> 126 3x3 11.3 -> 5.0 ms padded to 128; 1x1 1.7 -> 1.2 ms;
    130 -> 128 9.3 -> 6.4 ms with the input padded to 160; 258 -> 128 17.3 -> 10.1 ms). RAFT-style motion encoders
    (`128 - 2`), GRU inputs (`128 + 2 + ...`) and correlation features hit both. The padded-input gains are MLX
    switching to Winograd, so half precision outside kernels mode does not pad into it (see below).
    """
    cout, cin = wl.shape[0], wl.shape[3]
    if (
        _HALF_WINO_F32
        and (_HALF_WINO_F32 is True or (cin, cout) in _HALF_WINO_F32)
        and not _KERNELS
        and xl.dtype in (mx.float16, mx.bfloat16)
        and _mlx_winograd(wl.shape, cout, stride, dilation, groups, xl.shape[0] * xl.shape[1] * xl.shape[2])
    ):
        # MLX's half Winograd (relative mean 6e-3 against the direct conv's 3.5e-4) is too coarse for some nets
        f32 = _conv2d(xl.astype(mx.float32), wl.astype(mx.float32), stride, padding, dilation, groups)
        return f32.astype(xl.dtype)
    if ALIGN_OFF:
        return mx.conv2d(xl, wl, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if (
        not SPLIT_IN_OFF
        and groups == 1
        and wl.shape[1] * wl.shape[2] > 1
        and stride == (1, 1)
        and dilation == (1, 1)
        and xl.shape[1] * xl.shape[2] <= 48 * 48
        and (cin >= 1920 or (cin >= 1280 and xl.dtype == mx.float32 and xl.shape[1] * xl.shape[2] <= 24 * 24))
    ):
        # MLX's implicit GEMM slows down on very wide inputs at small spatial sizes (a UNet's concatenated skips);
        # input slices of 640 summed are faster (M1 Pro 3x3, 24^2 2560 -> 1280: float16 26.5 -> 12.2 ms, float32
        # 60.4 -> 14.5 ms; 12^2: 6.9 -> 3.8 / 15.5 -> 5.6 ms; float32 24^2 1280 -> 1280 12.7 -> 7.4 ms)
        parts = [
            _conv2d(xl[..., s : s + 640], wl[..., s : s + 640], stride, padding, dilation, groups)
            for s in range(0, cin, 640)
        ]
        return functools.reduce(operator.add, parts)
    pad_out = (-cout) % 16 if groups == 1 and cout >= 16 else 0
    spatial = wl.shape[1] * wl.shape[2] > 1
    pad_in = (-cin) % 32 if groups == 1 and cin >= 64 and stride == (1, 1) and spatial else 0
    o = cout + pad_out
    if (
        not WIDE_OUT_OFF
        and groups == 1
        and spatial
        and stride == (1, 1)
        and dilation == (1, 1)
        and (o == 48 or (o == 32 and (cin >= 128 or (cin >= 64 and xl.dtype == mx.float32))))
    ):
        # a 3x3 with 32-48 outputs takes MLX's narrow tile, slower than the 64-wide one (512^2, 160 -> 32:
        # float16 16.5 -> 13.2 ms, float32 27.8 -> 15.6 ms; 64 -> 32 float32 11.2 -> 6.8 ms; 160 -> 48 float32
        # 21.5 -> 15.8 ms; at 32 -> 32 and float16 64 -> 32 the narrow one is as fast or faster): zero outputs to 64
        pad_out = 64 - cout
    if (pad_out or pad_in) and not _KERNELS and xl.dtype in (mx.float16, mx.bfloat16):
        # Half precision: padding must not move the conv into MLX's Winograd, whose float16 error is ~17x the direct
        # conv's (relative mean 6e-3 against 3.5e-4, bfloat16 4.8e-2; float32 4e-6). Its speed-up was all the input
        # padding gave: without Winograd a padded input is no faster or much slower (512^2, 144 -> 160 inputs: 4x).
        # HAT's 180-channel convs padded to 192 put hat-x4 float16 over its golden (mean 0.00058 -> 0.00092).
        # Kernels mode runs its own Winograd (F(4,3), 2.9e-3) anyway and keeps the padding.
        pad_in = 0
        hw = xl.shape[0] * xl.shape[1] * xl.shape[2]
        if _mlx_winograd(wl.shape, cout + pad_out, stride, dilation, groups, hw) and not _mlx_winograd(
            wl.shape, cout, stride, dilation, groups, hw
        ):
            pad_out = 0
    if pad_out or pad_in:
        wl = mx.pad(wl, [(0, pad_out), (0, 0), (0, 0), (0, pad_in)])
    if pad_in:
        xl = mx.pad(xl, [(0, 0), (0, 0), (0, 0), (0, pad_in)])
    y = mx.conv2d(xl, wl, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y[..., :cout] if pad_out else y


def _pool_windows(
    x: mx.array,
    k: Sequence[int],
    s: Sequence[int],
    pad: Sequence[int],
    fill: float,
    *,
    ceil: bool = False,
    extra_fill: float | None = None,
) -> mx.array:
    """NCHW -> strided windows `[N, C, OH, OW, kh, kw]` of the padded input (`extra_fill`: the ceil-mode overhang)."""
    extra = [0, 0]
    if ceil:
        for i in range(2):
            size = x.shape[2 + i] + 2 * pad[i] - k[i]
            out = -(-size // s[i]) + 1
            if (out - 1) * s[i] >= x.shape[2 + i] + pad[i]:  # the last window must start inside the input
                out -= 1
            extra[i] = max(0, (out - 1) * s[i] + k[i] - (x.shape[2 + i] + 2 * pad[i]))
    if extra_fill is not None and any(extra):
        x = mx.pad(x, [(0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])], constant_values=fill) if any(pad) else x
        x = mx.pad(x, [(0, 0), (0, 0), (0, extra[0]), (0, extra[1])], constant_values=extra_fill)
    elif any(pad) or any(extra):
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
    ceil: bool = False,
) -> tuple[mx.array, None]:
    if tuple(d) != (1, 1):
        msg = "max_pool2d with dilation"
        raise UnsupportedOp(msg)
    s = s or k
    if x.ndim == 3:  # unbatched [C, H, W]
        return _pool_windows(x[None], k, s, pad, -float("inf"), ceil=ceil).max(axis=(-2, -1))[0], None
    return _pool_windows(x, k, s, pad, -float("inf"), ceil=ceil).max(axis=(-2, -1)), None


def _window_sum(x: mx.array, k: Sequence[int], s: Sequence[int], pad: Sequence[int]) -> mx.array:
    """Float32 window sums of NCHW `x` (floor mode) as k strided-slice adds per axis.

    The 6-D `as_strided` window view reduced over its last two axes ran 4 ms for a 3x3 / stride-2 pool of
    `[1, 128, 256, 376]` on an M1 Pro (RAFT-Stereo's `pool2x`, 32 per call); strided slice adds are a few
    elementwise kernels.

    Non-overlapping windows (`k == s`, the correlation pyramids) are one pass over the kh x kw strided views of the
    input, each cast to float32 on its own: compiled, one fused kernel writing only the pooled output. Casting the
    whole input first materialised a float32 copy of it: 4.2 GB for MemFlow's 1080p fp16 all-pairs volume (pool 2x2
    of `[8160, 1, 136, 240]` fp16: 762 -> 64 MB transient, 12.0 -> 2.2 ms). Overlapping windows keep the separable
    passes over a float32 copy: one pass over 9 views made Selective-IGEV's fp16 3x3 / stride-2 `pool2x` ~2.5% slower.
    """
    if tuple(k) == tuple(s) and not any(pad):
        oh, ow = x.shape[2] // k[0], x.shape[3] // k[1]
        acc = None
        for i in range(k[0]):
            for j in range(k[1]):
                part = x[:, :, i : i + k[0] * (oh - 1) + 1 : k[0], j : j + k[1] * (ow - 1) + 1 : k[1]]
                acc = part.astype(mx.float32) if acc is None else acc + part.astype(mx.float32)
        return acc
    y = x.astype(mx.float32)
    if any(pad):
        y = mx.pad(y, [(0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])])
    for axis, (kk, ss) in enumerate(zip(k, s, strict=True), start=2):
        n_out = (y.shape[axis] - kk) // ss + 1
        idx = [slice(None)] * 4
        acc = None
        for i in range(kk):
            idx[axis] = slice(i, i + ss * (n_out - 1) + 1, ss)
            part = y[tuple(idx)]
            acc = part if acc is None else acc + part
        y = acc
    return y


@_reg(aten.avg_pool2d.default)
def _avg_pool(
    x: mx.array,
    k: Sequence[int],
    s: Sequence[int] = (),
    pad: Sequence[int] = (0, 0),
    ceil: bool = False,
    count_include_pad: bool = True,
    divisor: int | None = None,
) -> mx.array:
    s = s or k
    if ceil or x.ndim != 4:
        win = _pool_windows(x, k, s, pad, 0.0, ceil=ceil).astype(mx.float32).sum(axis=(-2, -1))
    else:
        win = _window_sum(x, k, s, pad)
    if divisor is not None:
        win = win / divisor
    elif (any(pad) and not count_include_pad) or ceil:  # windows clipped to the input (and to the padding, if counted)
        ones = mx.ones((1, 1, *x.shape[2:]), mx.float32)
        fill = 1.0 if count_include_pad else 0.0
        win = win / _pool_windows(ones, k, s, pad, fill, ceil=ceil, extra_fill=0.0).sum(axis=(-2, -1))
    else:
        win = win / (k[0] * k[1])
    return win.astype(x.dtype)


@_reg(aten.avg_pool3d.default)
def _avg_pool3d(
    x: mx.array,
    k: Sequence[int],
    s: Sequence[int] = (),
    pad: Sequence[int] = (0, 0, 0),
    ceil: bool = False,
    count_include_pad: bool = True,
    divisor: int | None = None,
) -> mx.array:
    """Unpadded 3-D average pooling in float32: windows equal to the stride, or stride 1 (a sliding mean per axis)."""
    s = s or k
    if any(pad) or ceil or divisor is not None:
        msg = "avg_pool3d: only unpadded windows"
        raise UnsupportedOp(msg)
    if tuple(s) == (1, 1, 1):
        y = x.astype(mx.float32)
        for axis, kk in enumerate(k, start=2):
            if kk > 1:
                n_out = y.shape[axis] - kk + 1
                acc = mx.cumsum(y, axis=axis)
                head = mx.take(acc, mx.arange(kk - 1, kk - 1 + n_out), axis=axis)
                tail = mx.concatenate(
                    [
                        mx.zeros_like(mx.take(acc, mx.arange(1), axis=axis)),
                        mx.take(acc, mx.arange(n_out - 1), axis=axis),
                    ],
                    axis=axis,
                )
                y = (head - tail) / kk
        return y.astype(x.dtype)
    if tuple(s) != tuple(k):
        msg = "avg_pool3d: only windows equal to the stride or stride 1"
        raise UnsupportedOp(msg)
    n, c, d, h, w = x.shape
    od, oh, ow = d // k[0], h // k[1], w // k[2]
    y = x[:, :, : od * k[0], : oh * k[1], : ow * k[2]].astype(mx.float32)
    y = y.reshape(n, c, od, k[0], oh, k[1], ow, k[2]).mean(axis=(3, 5, 7))
    return y.astype(x.dtype)


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


@_reg(aten.adaptive_max_pool2d.default)
def _adaptive_max_pool(x: mx.array, size: Sequence[int]) -> tuple[mx.array, mx.array]:
    _, _, h, w = x.shape
    oh, ow = size
    rows = []
    for i in range(oh):
        h0, h1 = (i * h) // oh, -((-(i + 1) * h) // oh)
        cols = [x[:, :, h0:h1, (j * w) // ow : -((-(j + 1) * w) // ow)].max(axis=(2, 3)) for j in range(ow)]
        rows.append(mx.stack(cols, axis=-1))
    y = mx.stack(rows, axis=-2)
    return y, mx.zeros(y.shape, dtype=mx.int64)  # the indices are never read


@_reg(aten.constant_pad_nd.default)
def _pad(x: mx.array, pad: Sequence[int], value: float = 0.0) -> mx.array:
    widths = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):
        widths[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    return mx.pad(x, widths, constant_values=value)


_EDGE_PADS = (
    aten.reflection_pad1d.default,
    aten.reflection_pad2d.default,
    aten.reflection_pad3d.default,
    aten.replication_pad1d.default,
    aten.replication_pad2d.default,
    aten.replication_pad3d.default,
)
"""Kept from the export's decomposition: core ATen makes them index gathers (UniK3D's reflect-padded decoder convs,
[1, 64, 646, 952] float32: ~20 ms a gather, slices and a concatenation are memory-bound)."""


def _edge_pad(x: mx.array, pad: Sequence[int], *, reflect: bool) -> mx.array:
    """Reflection / replication padding of the trailing dims (`pad` in `F.pad` order) by slices and concatenations."""
    for i in range(len(pad) // 2):
        ax, lo, hi = x.ndim - 1 - i, pad[2 * i], pad[2 * i + 1]
        n = x.shape[ax]
        parts = []
        if lo > 0:
            idx = [slice(None)] * x.ndim
            idx[ax] = slice(lo, 0, -1) if reflect else slice(0, 1)
            part = x[tuple(idx)]
            parts.append(part if reflect else mx.repeat(part, lo, axis=ax))
        parts.append(x)
        if hi > 0:
            idx = [slice(None)] * x.ndim
            idx[ax] = slice(n - 2, (n - 2 - hi) if n - 2 - hi >= 0 else None, -1) if reflect else slice(n - 1, n)
            part = x[tuple(idx)]
            parts.append(part if reflect else mx.repeat(part, hi, axis=ax))
        x = mx.concatenate(parts, axis=ax) if len(parts) > 1 else x
    return x


def _reflect_pad(x: mx.array, pad: Sequence[int]) -> mx.array:
    return _edge_pad(x, pad, reflect=True)


def _replicate_pad(x: mx.array, pad: Sequence[int]) -> mx.array:
    return _edge_pad(x, pad, reflect=False)


for _op in _EDGE_PADS:
    OPS[_op] = _reflect_pad if "reflection" in str(_op) else _replicate_pad


def _src_index(out: int, inp: int, align: bool, scale: float | None) -> mx.array:
    i = mx.arange(out, dtype=mx.float32)
    if align:
        return i * ((inp - 1) / (out - 1)) if out > 1 else mx.zeros_like(i)
    ratio = (1.0 / scale) if scale else inp / out
    return mx.maximum((i + 0.5) * ratio - 0.5, 0.0)


@_reg(aten.upsample_bilinear2d.default, aten.upsample_bilinear2d.vec)
def _bilinear(x: mx.array, size: Sequence[int] | None, align: bool, *scales: Any) -> mx.array:
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
    if (
        x.ndim == 4
        and x.dtype in (mx.float16, mx.bfloat16, mx.float32)
        and not os.environ.get("FF_MLX_NO_FUSED_RESIZE")
    ):
        from fastforks.kernels.graph_mlx.ops import bilinear

        def ratio(out: int, inp: int, scale: float | None) -> float:
            if align:
                return (inp - 1) / (out - 1) if out > 1 else 0.0
            return (1.0 / scale) if scale else inp / out

        return bilinear(x, (oh, ow), (ratio(oh, h, sh), ratio(ow, w, sw)), align=align)
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


@_reg(aten.upsample_trilinear3d.vec)
def _trilinear(x: mx.array, size: Sequence[int] | None, align: bool, scales: Sequence[float] | None = None) -> mx.array:
    from fastforks.kernels.graph_mlx.ops import trilinear

    shape = x.shape[2:]
    out = tuple(size) if size is not None else tuple(int(s * f) for s, f in zip(shape, scales, strict=True))

    def ratio(o: int, i: int, f: float | None) -> float:
        if align:
            return (i - 1) / (o - 1) if o > 1 else 0.0
        return (1.0 / f) if (f and size is None) else i / o

    fs = scales if scales is not None else (None,) * 3
    return trilinear(x, out, tuple(ratio(o, i, f) for o, i, f in zip(out, shape, fs, strict=True)), align=align)  # type: ignore[arg-type]


def _cubic(t: np.ndarray) -> np.ndarray:
    """PyTorch's bicubic filter (a = -0.5)."""
    t = np.abs(t)
    near = ((1.5 * t - 2.5) * t) * t + 1.0
    far = (((-0.5 * t + 2.5) * t) - 4.0) * t + 2.0
    return np.where(t < 1.0, near, np.where(t < 2.0, far, 0.0))


def _aa_matrix(inp: int, out: int, align: bool, scale: float | None, *, cubic: bool = False) -> np.ndarray:
    """`[out, inp]` weights of PyTorch's antialiased bilinear / bicubic resize along one axis (normalised)."""
    s = (inp - 1) / (out - 1) if align and out > 1 else (1.0 / scale if scale else inp / out)
    support = max(s, 1.0) * (2.0 if cubic else 1.0)
    inv = 1.0 / s if s >= 1.0 else 1.0
    m = np.zeros((out, inp), dtype=np.float32)
    for i in range(out):
        c = s * i if align else s * (i + 0.5)
        lo = max(int(c - support + 0.5), 0)
        hi = min(int(c + support + 0.5), inp)
        t = (np.arange(lo, hi) - c + (0.0 if align else 0.5)) * inv
        w = _cubic(t) if cubic else np.maximum(0.0, 1.0 - np.abs(t))
        m[i, lo:hi] = w / w.sum() if w.sum() > 0 else 0.0
    return m


@_reg(aten._upsample_bilinear2d_aa.default, aten._upsample_bilinear2d_aa.vec)
def _bilinear_aa(x: mx.array, size: Sequence[int] | None, align: bool, *scales: Any) -> mx.array:
    return _resize_aa(x, size, align, scales, cubic=False)


@_reg(aten._upsample_bicubic2d_aa.default, aten._upsample_bicubic2d_aa.vec)
def _bicubic_aa(x: mx.array, size: Sequence[int] | None, align: bool, *scales: Any) -> mx.array:
    return _resize_aa(x, size, align, scales, cubic=True)


def _resize_aa(
    x: mx.array, size: Sequence[int] | None, align: bool, scales: tuple[Any, ...], *, cubic: bool
) -> mx.array:
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
    integer = x.dtype in (mx.uint8, mx.int8, mx.int16, mx.int32)
    xd = x.astype(mx.float32) if integer else x
    wy = mx.array(_aa_matrix(h, oh, align, sh, cubic=cubic)).astype(xd.dtype)
    wx = mx.array(_aa_matrix(w, ow, align, sw, cubic=cubic)).astype(xd.dtype)
    out = mx.matmul(mx.matmul(wy, xd), wx.T)
    if integer:  # torch rounds an integer resize and saturates it
        info = {mx.uint8: (0, 255), mx.int8: (-128, 127), mx.int16: (-32768, 32767), mx.int32: (-(2**31), 2**31 - 1)}[
            x.dtype
        ]
        out = mx.clip(mx.round(out), *info).astype(x.dtype)
    return out


@_reg(aten.upsample_nearest2d.default, aten.upsample_nearest2d.vec)
def _nearest(x: mx.array, size: Sequence[int] | None, *scales: Any) -> mx.array:
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


# ---- channels-last execution of layout-insensitive ops --------------------------------------------------------------
# Convolutions run NHWC and hand back `_nchw(y)`, a transposed *view*: the data stays NHWC in memory. An elementwise op,
# a concatenation or a pad that reads such a view in NCHW order gathers with a strided kernel (3-8x slower than a
# contiguous one) and writes NCHW, which the next convolution has to copy back to NHWC. `GraphNet` therefore tracks
# which values are such views (`_channels_last_values`) and runs the ops of `CL_OPS` on their channels-last views,
# returning channels-last results as NCHW views again. (Values that are really NCHW in memory keep the plain ops: a
# channels-last view of them would only move the strided access.)
CL_OPS: dict[Any, Callable[..., Any]] = {}
CL_OPS[torch.ops.fastforks.run_dtype.default] = _run_dtype  # a cast keeps the layout
CL_OPS[torch.ops.fastforks.float32.default] = _float32
_CL_ND = (3, 4, 5)


def _cl_perm(nd: int) -> tuple[int, ...]:
    return (0, *range(2, nd), 1)


def _cl(x: mx.array) -> mx.array:
    """The channels-last view (`[N, ..., C]`) of a `[N, C, ...]` array."""
    return mx.transpose(x, _cl_perm(x.ndim))


def _uncl(x: mx.array) -> mx.array:
    """The `[N, C, ...]` view of a channels-last array."""
    return mx.transpose(x, (0, x.ndim - 1, *range(1, x.ndim - 1)))


def _cl_axis(nd: int, axis: int) -> int:
    """Axis of the channels-last array that is axis `axis` of the `[N, C, ...]` one."""
    return _cl_perm(nd).index(axis % nd)


def _channels_last(f: Callable[..., Any]) -> Callable[..., Any]:
    """`f` (elementwise, arrays broadcast against each other) computed on channels-last views of its arrays.

    Applies when every array operand with rank > 0 has the same rank (3, 4 or 5): the views then broadcast exactly as
    the originals did. Otherwise `f` runs as it is.
    """

    @functools.wraps(f)
    def g(*args: Any, **kwargs: Any) -> Any:
        arrs = [a for a in args if isinstance(a, mx.array) and a.ndim]
        if (
            not arrs
            or arrs[0].ndim not in _CL_ND
            or any(a.ndim != arrs[0].ndim for a in arrs)
            or any(isinstance(v, mx.array) and v.ndim for v in kwargs.values())
        ):
            return f(*args, **kwargs)
        y = f(*[_cl(a) if isinstance(a, mx.array) and a.ndim else a for a in args], **kwargs)
        return _uncl(y) if isinstance(y, mx.array) and y.ndim == arrs[0].ndim else y

    return g


CL_ELEMENTWISE = _a(
    *(
        f"{n}.{o}"
        for n in (
            "add", "sub", "rsub", "mul", "div", "pow", "where", "masked_fill", "clamp", "clamp_min", "clamp_max",
            "eq", "ne", "lt", "le", "gt", "ge", "bitwise_and", "bitwise_or", "bitwise_xor", "remainder",
        )
        for o in ("Tensor", "Scalar", "self", "ScalarOther", "ScalarSelf", "Tensor_Tensor", "Tensor_Scalar")
    ),
    "neg", "abs", "exp", "log", "log1p", "expm1", "sqrt", "rsqrt", "sin", "cos", "tanh", "sigmoid", "erf", "floor",
    "ceil", "round", "reciprocal", "sign", "square", "relu", "silu", "hardtanh", "hardsigmoid", "hardswish",
    "leaky_relu", "elu", "softplus", "maximum", "minimum", "logical_not", "logical_and", "logical_or", "bitwise_not",
    "isnan", "gelu", "_to_copy", "to.dtype", "to.dtype_layout", "type_as",
)  # fmt: skip
for _t in CL_ELEMENTWISE:
    if _t in OPS:
        CL_OPS[_t] = _channels_last(OPS[_t])


@_reg_cl(aten.cat.default)
def _cat_cl(xs: Sequence[mx.array], d: int = 0) -> mx.array:
    """`aten.cat` along the channel (or a spatial) axis of channels-last views, when all operands are 3-5-D."""
    xs = list(xs)
    nd = xs[0].ndim
    if nd not in _CL_ND or d % nd == 0 or any(x.ndim != nd for x in xs):
        return mx.concatenate(xs, axis=d)
    return _uncl(mx.concatenate([_cl(x) for x in xs], axis=_cl_axis(nd, d)))


@_reg_cl(aten.upsample_nearest2d.default, aten.upsample_nearest2d.vec)
def _nearest_cl(x: mx.array, size: Sequence[int] | None, *scales: Any) -> mx.array:
    """`_nearest` on the channels-last view (4-D): an integer factor is a broadcast copy, anything else a gather."""
    if x.ndim != 4:
        return _nearest(x, size, *scales)
    h, w = x.shape[2:]
    if size is None:
        fac = scales[0]
        sh, sw = (fac[0], fac[1]) if isinstance(fac, (list, tuple)) else (scales[0], scales[1])
        oh, ow = int(h * sh), int(w * sw)
    else:
        oh, ow = size
    xl = _cl(x)
    if oh % h or ow % w:
        ys = mx.minimum(mx.floor(mx.arange(oh, dtype=mx.float32) * (h / oh)).astype(mx.int32), h - 1)
        xs = mx.minimum(mx.floor(mx.arange(ow, dtype=mx.float32) * (w / ow)).astype(mx.int32), w - 1)
        return _uncl(mx.take(mx.take(xl, ys, axis=1), xs, axis=2))
    fh, fw = oh // h, ow // w
    n, _, _, c = xl.shape
    y = mx.broadcast_to(xl[:, :, None, :, None, :], (n, h, fh, w, fw, c))
    return _uncl(y.reshape(n, oh, ow, c))


@_reg_cl(aten.constant_pad_nd.default)
def _pad_cl(x: mx.array, pad: Sequence[int], value: float = 0.0) -> mx.array:
    """`_pad` on the channels-last view (spatial axes only; a pad of the channel or batch axis stays as it was)."""
    nd = x.ndim
    if nd not in _CL_ND or len(pad) // 2 >= nd - 1:
        return _pad(x, pad, value)
    return _uncl(_pad(_cl(x), (0, 0, *pad), value))


@_reg_cl(aten.var_mean.correction)
def _var_mean_cl(
    x: mx.array, d: Sequence[int] | None = None, correction: float = 1, keepdim: bool = False
) -> tuple[mx.array, mx.array]:
    """`_var_mean` over the spatial axes of a channels-last view (an instance norm's statistics).

    A column reduction of the `[N, voxels, C]` rows the convolution wrote instead of a strided one in NCHW order
    (32 x 224^3 float16: 78 -> 44 ms); kernels mode: the one-pass `channel_stats` kernel (9 ms).
    """
    nd = x.ndim
    if nd not in _CL_ND or not d or sorted(a % nd for a in d) != list(range(2, nd)):
        return _var_mean(x, d, correction, keepdim)
    n, c = x.shape[:2]
    flat = _cl(x).reshape(n, -1, c)
    rows, ddof = flat.shape[1], int(correction or 0)
    if _KERNELS and rows >= 4096:
        from fastforks.kernels.graph_mlx import ops

        stats = [ops.channel_stats(flat[i]) for i in range(n)]
        mean = mx.stack([s[0] for s in stats])
        var = mx.stack([s[1] for s in stats]) * (rows / max(rows - ddof, 1))
    else:
        y = flat.astype(mx.float32)
        mean, var = mx.mean(y, axis=1), mx.var(y, axis=1, ddof=ddof)
    shape = (n, c, *(1,) * (nd - 2)) if keepdim else (n, c)
    return var.astype(x.dtype).reshape(shape), mean.astype(x.dtype).reshape(shape)


# ops that return a view of their (first) argument: a channels-last view stays one
_CL_VIEWS = (aten.slice.Tensor, aten.alias.default, aten.detach.default, aten.clone.default, aten.contiguous.default)


def _channels_last_values(nodes: list[Any], skips: dict[str, str]) -> set[str]:
    """Names of the graph values that are channels-last views: convolution outputs and what is computed from them.

    `skips`: value -> value it aliases (folded batch norm, fused activation).
    """
    tagged: set[str] = set()
    for node in nodes:
        if node.op != "call_function":
            continue
        if node.name in skips:
            if skips[node.name] in tagged:
                tagged.add(node.name)
            continue
        val = node.meta.get("val")
        srcs = list(_refs(node.args))
        if node.target is aten.var_mean.correction:  # statistics of a channels-last map: tagged, its outputs are not
            if srcs and srcs[0] in tagged:
                tagged.add(node.name)
            continue
        if not isinstance(val, torch.Tensor) or val.ndim not in _CL_ND:
            continue
        if (
            node.target is aten.convolution.default
            or (node.target in CL_OPS and any(r in tagged for r in srcs))
            or (node.target in _CL_VIEWS and srcs and srcs[0] in tagged)
        ):
            tagged.add(node.name)
    return tagged


# ---- interpreter ------------------------------------------------------------------------------------------------
_HAND_BACK_MIN = 1 << 20
_MADV_FREE_REUSABLE = 7  # macOS: the pages leave the footprint at once (the allocator would keep them resident)


def _exclusive(storage: torch.UntypedStorage) -> bool:
    """No tensor but one shares this storage (its use count: that tensor and this wrapper)."""
    return torch._C._storage_Use_Count(storage._cdata) == 2


def _hand_back(storage: torch.UntypedStorage) -> None:
    """Return the whole pages of a storage to the OS (a freed CPU twin stays in the footprint otherwise).

    Only for a storage that was `_exclusive` and is about to go: its contents are gone afterwards.
    """
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        page = mmap.PAGESIZE
        start = (storage.data_ptr() + page - 1) // page * page
        end = (storage.data_ptr() + storage.nbytes()) // page * page
        if end > start:
            libc.madvise(start, end - start, _MADV_FREE_REUSABLE)
    except (OSError, AttributeError):
        pass


# `FF_MLX_NO_PREFETCH=1`: no read-ahead of the weights during the export (a lazy twin's parameters are the mmapped
# checkpoint, so the first call's translation faults them in page by page otherwise: 11-24 s of vggt-1b's first call)
PREFETCH_OFF = bool(os.environ.get("FF_MLX_NO_PREFETCH"))
_PREFETCH_THREADS = 8  # parallel page faults: one thread reads a cold mmap at ~0.2 GB/s, several overlap the SSD reads
_PREFETCH_PIECE = 64 << 20  # bytes per touch (a thread lets go of a storage between pieces)
_PREFETCH_MIN = 64 << 20  # smaller nets: not worth the threads


def _prefetch(tensors: Iterable[Any]) -> list[threading.Thread]:
    """Threads that read one byte per page of these CPU tensors (the module's weights, to run while the export traces).

    A no-op for weights already in memory and the disk read of a mmapped checkpoint otherwise (clean file-backed pages,
    which the translation reads anyway). Touched, not `MADV_WILLNEED`: macOS counts read-ahead pages in the process's
    physical footprint (vggt-1b +4.3 GB peak), faulted ones it does not. The bytes of the tensors, not their storages:
    a safetensors checkpoint is one storage for the whole file. The threads hold the storages, so the caller joins
    them before anything checks `_exclusive`.
    """
    spans: dict[int, tuple[torch.UntypedStorage, list[tuple[int, int]]]] = {}
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.is_meta or t.numel() == 0:
            continue
        s = t.untyped_storage()
        off = t.data_ptr() - s.data_ptr()
        extent = (sum((n - 1) * st for n, st in zip(t.shape, t.stride(), strict=True)) + 1) * t.element_size()
        spans.setdefault(s.data_ptr(), (s, []))[1].append((off, off + extent))
    pieces: list[tuple[torch.UntypedStorage, int, int]] = []  # (storage, start, end) of at most _PREFETCH_PIECE bytes
    for s, ranges in spans.values():
        ranges.sort()
        merged = [list(ranges[0])]
        for a, b in ranges[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        pieces += [(s, a, min(a + _PREFETCH_PIECE, hi)) for lo, hi in merged for a in range(lo, hi, _PREFETCH_PIECE)]
    if sum(b - a for _, a, b in pieces) < _PREFETCH_MIN:
        return []
    page = mmap.PAGESIZE

    def touch(work: list[tuple[torch.UntypedStorage, int, int]]) -> None:
        with torch.no_grad():
            while work:
                s, a, b = work.pop()
                torch.empty(0, dtype=torch.uint8).set_(s, a, (b - a,))[::page].max()  # torch drops the GIL here
                del s

    n = _PREFETCH_THREADS  # contiguous runs of pieces per thread: sequential reads within each
    threads = [
        threading.Thread(target=touch, args=(pieces[i * len(pieces) // n : (i + 1) * len(pieces) // n][::-1],))
        for i in range(n)
    ]
    for th in threads:
        th.daemon = True
        th.start()
    return threads


def _plain(a: Any) -> Any:
    """`a` with its fx nodes replaced by their names (a plan must not keep the exported graph and its fake tensors)."""
    if hasattr(a, "name") and hasattr(a, "op"):
        return _Name(a.name)
    if isinstance(a, (list, tuple)):
        return type(a)(_plain(x) for x in a)
    if isinstance(a, dict):
        return {k: _plain(v) for k, v in a.items()}
    return a


def _refs(a: Any) -> Iterator[str]:
    """Names of the graph values (fx nodes, folded constants) inside an argument."""
    if isinstance(a, _Name) or (hasattr(a, "name") and hasattr(a, "op")):
        yield a.name
    elif isinstance(a, (list, tuple)):
        for x in a:
            yield from _refs(x)


CONST_FOLD_OFF = bool(os.environ.get("FF_MLX_NO_CONST_FOLD"))  # constant subgraphs run on every call
CONST_FOLD_MAX_MB = int(os.environ.get("FF_MLX_CONST_FOLD_MB", "512"))
"""Folded constants kept at most (else none are folded): their arrays stay resident for the net's lifetime."""


def _fold_constants(plan: list[tuple[Any, ...]], values: dict[str, Any]) -> list[tuple[Any, ...]]:
    """`plan` without the steps that read only constants: those run once here, their results become constants.

    Masks and coordinate grids that a port builds in the graph from the input's (static) shape — Swin's shifted
    window mask, recomputed by every shifted block — otherwise run on every call. Steps of the same op on the same
    arguments share one result (the 27 identical masks of SwinIR-L are one array).
    """
    if CONST_FOLD_OFF:
        return plan

    def canon(a: Any) -> Any:
        if isinstance(a, _Name):
            return keys[a.name]
        if isinstance(a, (list, tuple)):
            return tuple(canon(x) for x in a)
        return a

    def get(a: Any) -> Any:
        if isinstance(a, _Name):
            return env[a.name]
        if isinstance(a, (list, tuple)):
            return type(a)(get(x) for x in a)
        return a

    env: dict[str, Any] = dict(values)
    seen: dict[Any, int] = {}  # structural key (op, argument ids) -> id
    first: dict[int, str] = {}  # id -> the step that computed it
    ids = itertools.count()
    keys: dict[str, int] = {}  # constant name -> id of its structural key
    for n, a in values.items():  # the lifted scalars of each block (`_tensor_constant`s) by value
        small = isinstance(a, mx.array) and a.size <= 64
        k = ("value", a.dtype, tuple(a.shape), tuple(a.reshape(-1).tolist())) if small else ("const", n)
        keys[n] = seen.setdefault(k, next(ids))
    folded: set[str] = set()
    for p in plan:
        if p[0] == "alias" and p[2] in keys:
            keys[p[1]], env[p[1]] = keys[p[2]], env[p[2]]
            folded.add(p[1])
            continue
        if p[0] != "call" or getattr(p[2], "uncompilable", False):
            continue
        refs = [n for a in (*p[3], *p[4].values()) for n in _refs(a)]
        if not all(r in keys for r in refs):
            continue
        op = p[2]
        if isinstance(op, functools.partial):  # built per node (`_run_dtype`, `_cat_in`): the same op by value
            op = (op.func, op.args, tuple(sorted(op.keywords.items())))
        key: Any = (op, canon(p[3]), tuple(sorted((k, canon(v)) for k, v in p[4].items())))
        try:
            hash(key)
        except TypeError:
            key = ("step", p[1])
        if key in seen:
            env[p[1]] = env[first[seen[key]]]
        else:
            try:
                env[p[1]] = p[2](*get(p[3]), **get(p[4]))
            except (RuntimeError, ValueError, TypeError, IndexError, UnsupportedOp):  # runs only in the graph: kept
                continue
            seen[key] = next(ids)
            first[seen[key]] = p[1]
        keys[p[1]] = seen[key]
        folded.add(p[1])
    if not folded:
        return plan
    rest = [p for p in plan if p[0] not in ("call", "alias") or p[1] not in folded]
    need = {n for p in rest if p[0] == "call" for a in (*p[3], *p[4].values()) for n in _refs(a)}
    need.update(p[2] for p in rest if p[0] == "alias")
    need.update(n for p in rest if p[0] == "output" for n in _refs(p[1]))
    frontier = {n: env[n] for n in folded & need}
    env.clear()  # the folded steps' intermediates outlived the call otherwise (vitmatte-b fp16: 1.1 GB resident)
    arrays = {id(a): a for a in frontier.values() if isinstance(a, mx.array)}
    if any(not isinstance(a, (mx.array, int, float, bool)) for a in frontier.values()) or (
        sum(a.nbytes for a in arrays.values()) > CONST_FOLD_MAX_MB << 20
    ):
        return plan
    mx.eval(list(arrays.values()))
    for n in [n for n in values if n not in need]:  # constants only the folded steps read
        del values[n]
    values.update(frontier)
    return rest


TUNE_MIN_MB = 16
TUNE_SLACK = 1.03
"""A compiled segment with an activation above `TUNE_MIN_MB` runs once eagerly and once compiled on its first call and
keeps the compiled form unless it takes more than `TUNE_SLACK` times as long or peaks 25 % (and 64 MB) higher."""
SEG_COMPILE_MAX_MB = 200
"""A replay segment is `mx.compile`d only when none of its activations exceeds this many MB: compiling a segment with
huge maps (a 5-D cost volume) peaks at about twice the memory of the lazy replay."""

BIG_WEIGHTS = 1 << 30  # weights (run dtype) from which the checkpoint interval grows to the weights' size
BIG_CHUNK_MB = 2048
CHUNK_MB = 256  # lazy replay: evaluate the live values after this much new activation (see `GraphNet._runner`)

_CONSUMED = "\0consumed"  # key of the set of twin tensor names whose values live on MLX (in the shared constants)
# key of the tuning memo in the shared constants: a segment's structure (its ops, its biggest activation to a power of
# two) -> compiled won. A repeated block (the 24 layers of a ViT, every Swin stage) and the same segment at another
# input size of the same module reuse the first trial's decision instead of timing it again.
_TUNE_MEMO = "\0tune"


def _op_name(f: Any) -> str:
    """A plan step's operation by name (a `functools.partial` by its function's)."""
    f = getattr(f, "func", f)
    return getattr(f, "__qualname__", None) or type(f).__qualname__


def _tensor_dtype(dtype: mx.Dtype, t: torch.Tensor) -> mx.Dtype | None:
    return dtype if t.is_floating_point() else None


TUNE_LOG = bool(os.environ.get("FF_MLX_TUNE_LOG"))  # print each tuning trial (segment, times, peaks, decision)
PROFILE: dict[tuple[str, str], list[float]] | None = {} if os.environ.get("FF_MLX_PROFILE") else None
# `FF_MLX_COLD_LOG=1`: where a first call's time goes (export, translation, each segment's first against its fifth
# call by kind: eager, compiled, tuned, memo), printed after the fifth call of each input signature
COLD_LOG = bool(os.environ.get("FF_MLX_COLD_LOG"))


def _profiled(
    env: dict[str, Any], name: str, f: Callable[..., Any], args: Any, kwargs: Any, get: Callable[..., Any]
) -> None:
    """Profiling (`FF_MLX_PROFILE=1`): one op evaluated on its own; time and call count per op and output shape."""
    a = [get(env, x) for x in args]
    kw = {k: get(env, v) for k, v in kwargs.items()}
    mx.eval([x for x in a if isinstance(x, mx.array)])
    t0 = time.perf_counter()
    out = f(*a, **kw)
    mx.eval(out)
    dt = time.perf_counter() - t0
    env[name] = out
    first = out[0] if isinstance(out, (tuple, list)) and out else out
    shape = tuple(first.shape) if isinstance(first, mx.array) else ()
    ins = ",".join(str(tuple(x.shape)) for x in a[:2] if isinstance(x, mx.array))
    op = getattr(f, "__name__", str(f))
    if op == "<lambda>":  # the op table's lambdas: named by their line in this file
        op = f"lambda@{f.__code__.co_firstlineno}"
    rec = PROFILE.setdefault((op, f"{ins}->{shape}"), [0.0, 0])  # type: ignore[union-attr]
    rec[0] += dt
    rec[1] += 1


@contextlib.contextmanager
def _no_op_regions() -> Iterator[None]:
    """While exporting: autocast and grad-mode regions that change nothing are not recorded.

    `torch.export` records every `torch.autocast` and `torch.no_grad` region of the forward and then splits the graph at
    each one, regenerating the Python code of the whole graph per split (quadratic: IGEV's 65 autocast regions over 8k
    nodes are 10 to 20 s of a 29 s export). A region that leaves the state as it was (an autocast that is or turns off,
    for a device whose autocast is already off, such as a CUDA autocast on a Mac; `no_grad` under `no_grad`) has no
    effect on the traced ops, so it is skipped. A region that changes the state is recorded as before.
    """
    autocast = torch.amp.autocast_mode.autocast
    enter, exit_ = autocast.__enter__, autocast.__exit__
    set_grad = torch._C._set_grad_enabled

    def lean_enter(self: Any) -> Any:
        skip = not getattr(self, "_enabled", True) and not torch.is_autocast_enabled(self.device)
        self.__dict__.setdefault("_ff_skipped", []).append(skip)
        return self if skip else enter(self)

    def lean_exit(self: Any, *exc: object) -> Any:
        skipped = self.__dict__.get("_ff_skipped")
        return False if skipped and skipped.pop() else exit_(self, *exc)

    def lean_set_grad(mode: bool) -> None:
        if bool(mode) != torch.is_grad_enabled():
            set_grad(mode)

    autocast.__enter__, autocast.__exit__ = lean_enter, lean_exit  # type: ignore[method-assign]
    torch._C._set_grad_enabled = lean_set_grad
    try:
        yield
    finally:
        autocast.__enter__, autocast.__exit__ = enter, exit_  # type: ignore[method-assign]
        torch._C._set_grad_enabled = set_grad


TWO_PASS_EXPORT = bool(os.environ.get("FF_MLX_TWO_PASS_EXPORT"))
"""`FF_MLX_TWO_PASS_EXPORT=1`: export the training IR, then retrace it with `run_decompositions` (the public API path;
about twice the trace time of `_export_decomposed`)."""


@contextlib.contextmanager
def _no_stack_traces() -> Iterator[None]:
    """While exporting: no Python stack trace per graph node (nothing here reads them; 10 to 20 % of a trace)."""
    from torch.fx import config

    old, config.do_not_emit_stack_traces = config.do_not_emit_stack_traces, True
    try:
        yield
    finally:
        config.do_not_emit_stack_traces = old


@functools.cache
def _decomp_table() -> dict[Any, Any]:
    """`torch.export.default_decompositions()` minus the ops the translator runs natively (built once per process)."""
    decomp = torch.export.default_decompositions()
    for keep in (
        aten.scaled_dot_product_attention.default,
        aten.upsample_bilinear2d.vec,
        aten.upsample_trilinear3d.vec,
        aten.linalg_vector_norm.default,
        aten.roll.default,  # `mx.roll`: the decomposition is an index gather (several times slower)
        *_EDGE_PADS,
    ):
        decomp.pop(keep, None)
    return decomp


@functools.cache
def _split_decomp_table() -> tuple[dict[Any, Any], dict[Any, Any]]:
    """`_decomp_table` split as `run_decompositions` splits it: (composite ops, AOT tracer table); 40 ms a call."""
    from torch.export.exported_program import _split_decomp_table_to_cia_and_python_decomp

    return _split_decomp_table_to_cia_and_python_decomp(dict(_decomp_table()))


def _export_decomposed(module: torch.nn.Module, args: tuple[Any, ...]) -> Any:
    """`module` exported at `args` straight to the ATen opset of `decomp` in one trace, or None when that fails.

    `torch.export.export` traces the pre-dispatch training IR and `run_decompositions` traces that graph again, which
    on an unrolled GRU loop (RAFT-Stereo's 32 updates, 6k nodes) is more than half of the export. This is the
    post-dispatch export (`pre_dispatch=False`, the pre-2.5 default) with `run_decompositions`' own table split: the
    composite (CIA) ops to keep or replace are overridden while tracing, the others go to the AOT tracer's table. The
    program has the same ops and signature, minus the no-op `_assert_tensor_metadata` nodes. Private torch API: any
    failure (including a lazily built buffer that export must not assign) returns None for the two-pass path.
    """
    try:
        from torch.export import _trace
        from torch.export.exported_program import _override_composite_implicit_decomp

        cia, python_decomp = _split_decomp_table()
        to_aten = _trace._export_to_aten_ir
        _trace._export_to_aten_ir = functools.partial(to_aten, decomp_table=python_decomp)
        try:
            with _override_composite_implicit_decomp(cia):
                return _trace._export(module, args, strict=False, pre_dispatch=False)
        finally:
            _trace._export_to_aten_ir = to_aten
    except Exception:  # noqa: BLE001 - private API or a module only the two-pass path exports: take that path
        return None


def _used_tensors(module: torch.nn.Module, args: tuple[Any, ...]) -> set[int] | None:
    """Ids of the parameters / buffers a forward at `args` passes to a torch function (a fake-tensor run), or None."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.overrides import TorchFunctionMode

    ids = {id(t) for _, t in _tensors(module)}
    seen: set[int] = set()

    class _Record(TorchFunctionMode):
        def __torch_function__(self, func: Any, types: Any, a: Any = (), kw: Any = None) -> Any:
            kw = kw or {}
            seen.update(id(x) for x in pytree.tree_leaves((a, kw)) if isinstance(x, torch.Tensor) and id(x) in ids)
            return func(*a, **kw)

    try:
        with FakeTensorMode(allow_non_fake_inputs=True) as fake, _Record():
            module(*(fake.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in args))
    except Exception:  # noqa: BLE001 - only a shortcut: the full export runs instead
        return None
    return seen


@contextlib.contextmanager
def _unused_hidden(module: torch.nn.Module, used: set[int]) -> Iterator[dict[str, torch.Tensor]]:
    """While exporting: the parameters / buffers outside `used` as plain attributes, which export does not lift.

    A method of a large net (a runner's `encode_image`) reads a tenth of its weights, and each lifted placeholder
    costs about 0.5 ms of fakifying and functionalizing. A tensor read that the fake run missed still works (export
    lifts it as a constant: `"a.lifted_tensor_0"`, on the hidden tensor's storage); the caller checks the program's
    constants against the yielded tensors (by name) and exports again in full.
    """
    hidden: dict[str, torch.Tensor] = {}
    saved: list[tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]] = []
    for prefix, host in module.named_modules(remove_duplicate=False):
        drop = {
            n: t for n, t in (*host._parameters.items(), *host._buffers.items()) if t is not None and id(t) not in used
        }
        if not drop:
            continue
        saved.append((host, host._parameters, host._buffers))
        host._parameters = {n: t for n, t in host._parameters.items() if n not in drop}
        host._buffers = {n: t for n, t in host._buffers.items() if n not in drop}
        for n, t in drop.items():
            host.__dict__[n] = t
            hidden[f"{prefix}.{n}" if prefix else n] = t
    try:
        yield hidden
    finally:
        for host, params, buffers in saved:
            for n in (*params, *buffers):
                host.__dict__.pop(n, None)
            host._parameters, host._buffers = params, buffers


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
        chunk_mb: int = 0,
        pipeline: bool = False,
        half_wino_f32: bool | tuple[tuple[int, int], ...] = False,
    ) -> None:
        """Keep the module until its first export; parameters convert then (`chunk_mb`: see `_runner`, 0 = `CHUNK_MB`).

        `half_wino_f32=True`: in float16 / bfloat16, the 3x3 convs MLX would run as Winograd run in float32 instead;
        a tuple of `(cin, cout)` pairs limits that to those convs.

        `pipeline=True`: queue each checkpoint segment (`mx.async_eval`) and wait for the one before it, so the next
        segment's graph is built while the GPU runs (one more segment's activations in flight).
        """
        self.module = module.eval()
        for m in self.module.modules():  # a buffer computed from parameters (position biases) is a non-leaf tensor
            for name, b in m._buffers.items():
                if b is not None and b.requires_grad:
                    m._buffers[name] = b.detach()
        self.dtype = getattr(mx, dtype)
        self._compile = compile
        self._itemsize = self.dtype.size
        self._sizes: dict[str, int] = {}
        self.max_act = 0
        self.weight_bytes = 0
        self.kernels = kernels
        self.half_wino_f32 = half_wino_f32
        self.chunk_mb = chunk_mb
        self.pipeline = pipeline
        self._runs: dict[tuple[Any, ...], Callable[..., Any]] = {}
        self.on_built: Callable[[], None] | None = None  # called after each export, before its first run
        self._wino: dict[
            Any, tuple[mx.array, mx.array]
        ] = {}  # kernels mode: Winograd-domain weights of 3x3 convolutions, by (id(weight), tile)
        self._consts: dict[str, Any] = {} if shared is None else shared
        self._consumed: set[str] = self._consts.setdefault(_CONSUMED, set())  # twin tensors the MLX side has taken over
        self._tune_memo: dict[Any, bool] = self._consts.setdefault(_TUNE_MEMO, {})
        # on-disk plan cache (`mlx_plan_cache`, set by `MlxModule` when `FASTFORKS_MLX_PLAN_CACHE` is on): the entry of
        # this net's input signature, what to store beside the program and how to restore it on a hit
        self.plan: Any = None
        self.plan_extra: Callable[[], Any] | None = None
        self.plan_restore: Callable[[Any], None] | None = None
        self._cold: dict[str, Any] = {}  # `COLD_LOG`: the current build's phase times
        self._prefetched = False  # `_prefetch` ran once (a later signature's build finds the weights read)
        self._prefetching: list[threading.Thread] = []

    def _const(self, name: str, t: torch.Tensor, *, share: bool = True, transpose: bool = False) -> mx.array:
        """`t` on MLX in the run dtype (`transpose`: a 2-D tensor stored as its contiguous transpose)."""
        key = f"{name}\0T" if transpose else name
        if share and key in self._consts:
            self._consumed.add(name)
            return self._consts[key]
        a = to_mx(t)
        if t.is_floating_point():
            a = a.astype(self.dtype)
        if transpose:  # on the GPU (a CPU transpose of a ViT-L's weights adds 0.3 s to the first call)
            a = mx.contiguous(a.T)
        mx.eval(a)
        if share:
            self._consts[key] = a
            self._consumed.add(name)
        return a

    def _build(self, args: tuple[torch.Tensor, ...]) -> Callable[..., Any]:
        token = EXPORTING_FOR_MLX.set(True)
        t0 = time.perf_counter()
        if not PREFETCH_OFF and not self._prefetched:
            self._prefetched = True
            self._prefetching = _prefetch(itertools.chain(self.module.parameters(), self.module.buffers()))
        try:
            return self._build_graph(args)
        finally:
            EXPORTING_FOR_MLX.reset(token)
            if COLD_LOG:
                self._cold["build"] = time.perf_counter() - t0

    def _export(self, args: tuple[torch.Tensor, ...]) -> Any:
        """The module exported at `args` and decomposed to core ATen: from the plan cache when it has the entry."""
        t0 = time.perf_counter()
        try:
            return self._export_program(args)
        finally:
            if COLD_LOG:
                self._cold["export"] = time.perf_counter() - t0

    def _export_program(self, args: tuple[torch.Tensor, ...]) -> Any:
        if self.plan is not None and (got := self.plan.load_program(self.module)) is not None:
            ep, extra = got
            if self.plan_restore is not None:
                self.plan_restore(extra)
            return ep
        filled = False
        with torch.no_grad(), _no_op_regions(), _no_stack_traces():
            ep = None if TWO_PASS_EXPORT else self._export_used(args)
            if ep is None and not TWO_PASS_EXPORT:
                ep = _export_decomposed(self.module, args)
            if ep is None:
                try:
                    ep = torch.export.export(self.module, args, strict=False)
                except (RuntimeError, ValueError, AssertionError, UserWarning):
                    # lazily built buffers (attention bias caches) that export must not assign: fill them with one
                    # real forward (its CPU activations are the peak of the build, so only a module that needs it pays)
                    self.module(*args)
                    filled = True
                    ep = torch.export.export(self.module, args, strict=False)
                t0 = time.perf_counter()
                ep = ep.run_decompositions(dict(_decomp_table()))
                if COLD_LOG:
                    self._cold["decomp"] = time.perf_counter() - t0
        if self.plan is not None and not filled:  # a loaded program could not fill the lazy buffers: not cached
            self.plan.save_program(ep, self.module, self.plan_extra() if self.plan_extra is not None else None)
        return ep

    def _export_used(self, args: tuple[torch.Tensor, ...]) -> Any:
        """The one-pass export with only the tensors a fake run reads lifted (`_Method(prune=True)`), or None.

        None (the full export follows) when the module does not ask for it, the fake run fails, it reads most of the
        tensors anyway, or the export lifted a tensor the fake run missed.
        """
        inner = getattr(self.module, "module", None)
        if not getattr(inner, "prune_unused", False):
            return None
        modules = list(self.module.modules())
        state = [(m, dict(m.__dict__), dict(m._parameters), dict(m._buffers)) for m in modules]
        try:
            used = _used_tensors(self.module, args)
        finally:  # a forward that caches a tensor on the module (a lazy grid) must not keep the fake one
            for m, d, p, b in state:
                m.__dict__.clear()
                m.__dict__.update(d)
                m._parameters.clear()
                m._parameters.update(p)
                m._buffers.clear()
                m._buffers.update(b)
        total = sum(1 for _ in _tensors(self.module))
        if used is None or len(used) > total * 3 // 4:
            return None
        with _unused_hidden(self.module, used) as hidden:
            ep = _export_decomposed(self.module, args)
        if ep is None:
            return None
        ptrs = {p for t in hidden.values() if (p := t.untyped_storage().data_ptr())}
        missed = any(
            isinstance(c, torch.Tensor) and c.untyped_storage().data_ptr() in ptrs for c in ep.constants.values()
        )
        return None if missed or set(hidden) & {str(s.target) for s in ep.graph_signature.input_specs} else ep

    def _build_graph(self, args: tuple[torch.Tensor, ...]) -> Callable[..., Any]:
        ep = self._export(args)
        graph = ep.graph_module.graph
        sig = ep.graph_signature
        state = set(ep.state_dict)
        lifted = {**ep.state_dict, **dict(ep.constants)}
        if self._prefetched and not PREFETCH_OFF:  # tensors the export lifted that are no parameter (plain attributes)
            self._prefetching += _prefetch(
                v for k, v in lifted.items() if k not in self._consts and f"{k}\0T" not in self._consts
            )
        while self._prefetching:  # the weights are read next (and the threads must let go of their storages)
            self._prefetching.pop().join()
        names = {s.arg.name: s.target for s in sig.input_specs if s.target is not None}
        nodes = list(graph.nodes)
        plan: list[tuple[Any, ...]] = []
        values: dict[str, Any] = {}  # constants (parameters, buffers) by node name
        params: dict[str, str] = {}  # parameter / buffer placeholders: node name -> target
        folds = self._fold_batch_norms(nodes, names, lifted)
        fused = (
            self._fuse_activations(nodes, names, lifted, values)
            if self.kernels
            else type("Fused", (), {"skip": {}, "conv": {}, "res": {}})()
        )
        cl_values = (
            set() if os.environ.get("FF_MLX_NO_CL") else _channels_last_values(nodes, {**folds.skip, **fused.skip})
        )
        merged = (
            {}
            if MERGE_OFF
            else self._merge_siblings(nodes, names, lifted, values, {*folds.args, *fused.conv, *fused.res})
        )
        half_cats = _conv_only_cats(nodes) if self.dtype in (mx.float16, mx.bfloat16) and not KEEP_F32_CONV else set()
        mask_parts, mask_skip = _sdpa_mask_parts(nodes)
        transposed = _transposed_weights(nodes, names)
        for node in nodes:
            if node.op == "placeholder":
                tgt = names.get(node.name)
                if tgt is not None:
                    params[node.name] = str(tgt)
                else:
                    plan.append(("input", node.name))
            elif node.op == "call_function":
                if node.name in folds.skip:
                    plan.append(("alias", node.name, folds.skip[node.name]))
                    continue
                if node.name in fused.skip:
                    plan.append(("alias", node.name, fused.skip[node.name]))
                    continue
                if node.name in merged:
                    lead, lo, hi, wk, bk = merged[node.name]
                    if wk is not None:  # the first sibling runs the merged convolution
                        a = node.args
                        args = (a[0], _Name(wk), _Name(bk) if bk else None, *a[3:])
                        plan.append(("call", lead, _conv, _plain(args), {}))
                    plan.append(("call", node.name, _channel_slice, (_Name(lead), lo, hi), {}))
                    continue
                if node.name in fused.res:
                    wk, bk, rname = fused.res[node.name]
                    a = node.args
                    args = (a[0], _Name(wk), _Name(bk) if bk else None, *a[3:], _Name(rname))
                    plan.append(("call", node.name, _conv_res, _plain(args), {}))
                    continue
                if node.name in mask_skip:  # built by `_sdpa_parts` from its parts
                    continue
                if node.name in transposed:  # a constant: the weight stored transposed
                    continue
                if node.name in mask_parts:
                    fn, args = mask_parts[node.name]
                    plan.append(("call", node.name, fn, _plain(tuple(args)), _plain(dict(node.kwargs))))
                    continue
                if node.name in fused.conv:
                    act, slope = fused.conv[node.name]
                    kw = {"act": act, "slope": _Name(slope) if slope else None}
                    plan.append(("call", node.name, _conv_act, _plain(folds.args.get(node.name, node.args)), kw))
                    continue
                args = folds.args.get(node.name, node.args)
                op = CL_OPS[node.target] if node.name in cl_values and node.target in CL_OPS else self._op(node.target)
                if op is _run_dtype:
                    op = functools.partial(mx.astype, dtype=self.dtype)
                elif node.name in half_cats:
                    op = functools.partial(_cat_in, op, self.dtype)
                plan.append(("call", node.name, op, _plain(args), _plain(dict(node.kwargs))))
            elif node.op == "output":
                plan.append(("output", _plain(node.args[0])))
        # activation sizes in the run dtype (bytes): the largest decides whether compiling pays, the rest pace the
        # lazy replay's checkpoints
        self._sizes = {
            node.name: math.prod(int(d) for d in v.shape) * self._itemsize
            for node in nodes
            if node.op == "call_function" and isinstance(v := node.meta.get("val"), torch.Tensor)
        }
        for name, (lead, *_) in merged.items():  # a merged convolution's output: its siblings' together
            self._sizes[lead] = self._sizes.get(lead, 0) + self._sizes.get(name, 0)
        self.max_act = max(self._sizes.values(), default=0)
        self.weight_bytes = sum(t.numel() for t in lifted.values() if isinstance(t, torch.Tensor)) * self._itemsize
        values.update(folds.values)
        used = {n for p in plan if p[0] == "call" for a in (*p[3], *p[4].values()) for n in _refs(a)}
        used.update(n for p in plan if p[0] == "output" for n in _refs(p[1]))
        for node_name, tgt in params.items():  # only what a node reads: a folded convolution's own weight is not
            if node_name in used:
                values[node_name] = self._const(tgt, lifted[tgt], share=tgt in state)
        for node_name, tgt in transposed.items():
            if node_name in used:
                values[node_name] = self._const(tgt, lifted[tgt], share=tgt in state, transpose=True)
        if transposed:
            mx.clear_cache()  # the untransposed copies: they would stay in the buffer cache (fp16 CLIP-L: +58 MB peak)
        plan = _fold_constants(plan, values)
        if not os.environ.get("FF_MLX_NO_BAND"):
            from fastforks.shared.mlx_band import band_tail

            plan = band_tail(nodes, plan, values, cl_values, self._itemsize)
        report = (
            mlx_dtype_report.Report(mlx_dtype_report.meta(nodes), self.dtype)
            if mlx_dtype_report.ON and self.dtype != mx.float32
            else None
        )
        del ep, graph, sig, lifted, nodes
        if self.on_built is not None:  # the release that follows hands back only tensors the program no longer holds
            gc.collect()  # (60 to 90 ms: without a release the twin keeps its tensors, the program's cycles can wait)
        return self._runner(plan, values, report)

    @staticmethod
    def _op(target: Any) -> Callable[..., Any]:
        if target is operator.getitem:
            return operator.getitem
        try:
            return OPS[target]
        except KeyError:
            msg = f"no MLX mapping for {target}"
            raise UnsupportedOp(msg) from None

    def _merge_siblings(
        self,
        nodes: list[Any],
        names: dict[str, Any],
        lifted: dict[str, torch.Tensor],
        values: dict[str, Any],
        exclude: set[str],
    ) -> dict[str, tuple[str, int, int, str | None, str | None]]:
        """Convolutions that read the same value with the same geometry, merged into one along the output channels.

        A GRU's update and reset gates (`convz(hx)`, `convr(hx)`) and twin heads become one wider convolution whose
        output the siblings slice: the input is read (and, for Winograd, transformed) once and the GEMM is wider.
        Returns conv name -> (merged value, first channel, end channel, merged weight key, bias key); the keys are set
        on the first sibling only.
        """
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for conv in nodes:
            if conv.op != "call_function" or conv.target is not aten.convolution.default or conv.name in exclude:
                continue
            a = conv.args
            wv = a[1].meta.get("val") if hasattr(a[1], "meta") else None
            if a[6] or a[8] != 1 or wv is None or wv.ndim != 4 or names.get(a[1].name) is None:
                continue
            if a[2] is not None and names.get(a[2].name) is None:
                continue
            key = (a[0].name, tuple(wv.shape[1:]), tuple(a[3]), tuple(a[4]), tuple(a[5]), a[2] is None)
            groups.setdefault(key, []).append(conv)
        out: dict[str, tuple[str, int, int, str | None, str | None]] = {}
        for convs in groups.values():
            if len(convs) < 2:
                continue
            wnames = [names[c.args[1].name] for c in convs]
            bnames = [names[c.args[2].name] for c in convs] if convs[0].args[2] is not None else []
            lead = f"{convs[0].name}__merged"
            wk = "\0".join(wnames) + "\0merged"
            bk = "\0".join(bnames) + "\0merged" if bnames else None
            if wk not in self._consts:
                self._consts[wk] = self._const(f"{wk}\0w", torch.cat([lifted[n] for n in wnames]), share=False)
            values[wk] = self._consts[wk]
            if bk is not None:
                if bk not in self._consts:
                    self._consts[bk] = self._const(f"{bk}\0b", torch.cat([lifted[n] for n in bnames]), share=False)
                values[bk] = self._consts[bk]
            self._consumed.update((*wnames, *bnames))
            lo = 0
            for i, c in enumerate(convs):
                hi = lo + int(c.args[1].meta["val"].shape[0])
                out[c.name] = (lead, lo, hi, wk if i == 0 else None, bk if i == 0 else None)
                lo = hi
        return out

    def _fuse_activations(
        self, nodes: list[Any], names: dict[str, Any], lifted: dict[str, torch.Tensor], values: dict[str, Any]
    ) -> Any:
        """Kernels mode: a 3x3 convolution whose only consumer is ReLU / PReLU (gt, mul, where) applies it."""
        res = type("Fused", (), {"skip": {}, "conv": {}, "res": {}})()
        for conv in nodes:
            if conv.op != "call_function" or conv.target is not aten.convolution.default:
                continue
            wv = conv.args[1].meta.get("val")
            xv = conv.args[0].meta.get("val")
            if conv.args[6] or conv.args[8] != 1 or wv is None or xv is None or wv.ndim != 4:
                continue
            wino = (
                tuple(wv.shape[2:]) == (3, 3)
                and list(conv.args[3]) == [1, 1]
                and list(conv.args[4]) == [1, 1]
                and list(conv.args[5]) == [1, 1]
            )
            if not wino and not _direct_ok(xv.shape[1], xv.shape[2] * xv.shape[3], wv.shape, conv.args[3]):
                continue
            users = list(conv.users)
            if wino and self._fuse_residual(conv, users, nodes, names, lifted, values, res):
                continue
            if len(users) == 1 and users[0].target is aten.relu.default:
                res.conv[conv.name] = (1, None)
                res.skip[users[0].name] = conv.name
            elif len(users) == 1 and users[0].target is aten.leaky_relu.default:
                key = f"{conv.name}__slope"
                slope = users[0].args[1] if len(users[0].args) > 1 else 0.01
                values[key] = mx.full((wv.shape[0],), float(slope), dtype=mx.float32)
                res.conv[conv.name] = (2, key)
                res.skip[users[0].name] = conv.name
            elif len(users) == 3 and {u.target for u in users} == {
                aten.gt.Scalar,
                aten.mul.Tensor,
                aten.where.self,
            }:
                gt, mul, where = (
                    next(u for u in users if u.target is t) for t in (aten.gt.Scalar, aten.mul.Tensor, aten.where.self)
                )
                view = mul.args[0]
                pname = names.get(view.args[0].name) if getattr(view, "target", None) is aten.view.default else None
                if (
                    pname is None
                    or gt.args[1] != 0
                    or mul.args[1] is not conv
                    or tuple(where.args) != (gt, conv, mul)
                    or len(view.users) != 1
                    or lifted[pname].numel() not in (1, wv.shape[0])
                ):
                    continue
                key = f"{conv.name}__slope"
                values[key] = mx.broadcast_to(self._const(pname, lifted[pname]).reshape(-1), (wv.shape[0],))
                res.conv[conv.name] = (2, key)
                res.skip.update({gt.name: conv.name, mul.name: conv.name, where.name: conv.name})
        return res

    def _fuse_residual(
        self,
        conv: Any,
        users: list[Any],
        nodes: list[Any],
        names: dict[str, Any],
        lifted: dict[str, torch.Tensor],
        values: dict[str, Any],
        res: Any,
    ) -> bool:
        """`conv * c + r` (a residual-block tail): the scale folds into the weights and bias, `r` into the epilogue."""
        if len(users) != 1 or users[0].target is not aten.mul.Tensor or users[0].args[0] is not conv:
            return False
        mul = users[0]
        scale = mul.args[1]
        if not isinstance(scale, (int, float)) or len(mul.users) != 1:
            return False
        (add,) = mul.users
        if add.target is not aten.add.Tensor or add.kwargs or len(add.args) != 2:
            return False
        other = add.args[1] if add.args[0] is mul else add.args[0]
        if (
            getattr(other, "op", None) != "call_function"
            or nodes.index(other) > nodes.index(conv)
            or other.meta.get("val") is None
            or tuple(other.meta["val"].shape) != tuple(conv.meta["val"].shape)
        ):
            return False
        wname = names.get(conv.args[1].name)
        bname = names.get(conv.args[2].name) if conv.args[2] is not None else None
        if wname is None or (conv.args[2] is not None and bname is None):
            return False
        wk, bk = f"{conv.name}__w", f"{conv.name}__b"
        # by parameter names: another input shape reuses it, the twin may be gone
        key = f"{wname}\0{bname}\0{scale}\0scaled"
        if key not in self._consts:
            w2 = to_mx((lifted[wname].double() * scale).float()).astype(self.dtype)
            b2 = to_mx((lifted[bname].double() * scale).float()).astype(self.dtype) if bname is not None else None
            mx.eval(w2, *(() if b2 is None else (b2,)))
            self._consts[key] = (w2, b2)
        values[wk], b2 = self._consts[key]
        if b2 is not None:
            values[bk] = b2
        self._consumed.update(t for t in (wname, bname) if t)
        res.res[conv.name] = (wk, bk if bname is not None else None, other.name)
        res.skip[mul.name] = conv.name
        res.skip[add.name] = conv.name
        return True

    def _fold_batch_norms(self, nodes: list[Any], names: dict[str, Any], lifted: dict[str, torch.Tensor]) -> Any:
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
            wk, bk = f"{conv.name}__w", f"{conv.name}__b"
            key = (
                f"{wname}\0{params[2]}\0fold"  # by parameter names: another input shape reuses it, the twin may be gone
            )
            if key not in self._consts:
                eps = bn.args[-1]
                w = lifted[wname].double()
                gamma = lifted[params[0]].double() if params[0] else torch.ones(w.shape[0], dtype=torch.double)
                beta = lifted[params[1]].double() if params[1] else torch.zeros(w.shape[0], dtype=torch.double)
                scale = gamma / torch.sqrt(lifted[params[3]].double() + eps)
                bias0 = lifted[names[conv.args[2].name]].double() if conv.args[2] is not None else 0
                w2 = (w * scale.view(-1, *[1] * (w.ndim - 1))).float()
                b2 = ((bias0 - lifted[params[2]].double()) * scale + beta).float()
                self._consts[key] = (to_mx(w2).astype(self.dtype), to_mx(b2).astype(self.dtype))
                mx.eval(*self._consts[key])
            res.values[wk], res.values[bk] = self._consts[key]
            self._consumed.update(t for t in (wname, *params, names[conv.args[2].name] if conv.args[2] else None) if t)
            # the convolution reads the folded constants; the bn node and its getitem alias the convolution's output
            res.args[conv.name] = (conv.args[0], _Name(wk), _Name(bk), *conv.args[3:])
            res.skip[bn.name] = conv.name
            res.skip[users[0].name] = conv.name
        return res

    def _runner(
        self, plan: list[tuple[Any, ...]], values: dict[str, Any], report: mlx_dtype_report.Report | None = None
    ) -> Callable[..., Any]:
        """The replay as segments of about `CHUNK_MB` of new activations each, evaluated one after the other.

        MLX would otherwise schedule the whole lazy graph at the end and hold far more than the live set. A segment is
        a function of the values still needed from before it; with compiling on, each is `mx.compile`d (fusion stays,
        the live set is bounded).
        """
        out_names: Any = None
        for p in plan:
            if p[0] == "output":
                out_names = p[1]
        steps = [(i, p) for i, p in enumerate(plan) if p[0] in ("call", "alias")]
        uses: dict[int, set[str]] = {}
        last: dict[str, int] = {}  # plan step of a value's last use
        for i, p in steps:
            refs = {n for a in (*p[3], *p[4].values()) for n in _refs(a)} if p[0] == "call" else {p[2]}
            uses[i] = refs
            last.update(dict.fromkeys(refs, i))
        out_refs = set(_refs(out_names))
        # a net with big weights peaks while loading (the float32 twin), not while running: checkpoint rarely there
        big = min(self.weight_bytes, BIG_CHUNK_MB << 20) if self.weight_bytes >= BIG_WEIGHTS else 0
        # a declared interval (`chunk_mb`, or the sweep's `FF_MLX_CHUNK_MB`) is taken as it is
        declared = os.environ.get("FF_MLX_CHUNK_MB") or self.chunk_mb
        chunk = int(declared) << 20 if declared else max(CHUNK_MB << 20, big)
        cuts, pending = [0], 0
        for k, (_, p) in enumerate(steps):
            pending += self._sizes.get(p[1], 0)
            if pending > chunk and k + 1 < len(steps):
                cuts.append(k + 1)
                pending = 0
        for k, (_, p) in enumerate(steps):  # a banded tail evaluates band by band: a segment of its own, not compiled
            if getattr(p[2], "uncompilable", False):
                cuts.extend(c for c in (k, k + 1) if 0 < c < len(steps))
        cuts = sorted({*cuts})
        cuts.append(len(steps))
        compiled = self._compile
        seg_max = int(os.environ.get("FF_MLX_SEG_MB", SEG_COMPILE_MAX_MB)) << 20
        if os.environ.get("FF_MLX_COMPILE"):
            compiled = os.environ["FF_MLX_COMPILE"] == "1"
            seg_max = 1 << 60
        if PROFILE is not None or report is not None:
            compiled = False

        def get(env: dict[str, Any], a: Any) -> Any:
            if isinstance(a, _Name) or (hasattr(a, "name") and hasattr(a, "op")):
                return env[a.name]
            if isinstance(a, (list, tuple)):
                return type(a)(get(env, x) for x in a)
            return a

        bounds = [(a, b) for a, b in itertools.pairwise(cuts) if b > a]
        # the tuning decisions of an earlier process (plan cache): compiled (True) / eager (False) per segment
        tune_sig = (len(bounds), compiled, seg_max, self.pipeline)
        known = self.plan.load_tune(tune_sig) if self.plan is not None else None
        known = known if known is not None and len(known) == len(bounds) else [None] * len(bounds)
        decided: list[bool | None] = list(known)
        kinds = ["eager"] * len(bounds)  # `COLD_LOG`: how each segment's first call ran
        cold_runs: list[list[float]] = []

        def make(k: int, seg: list[tuple[int, tuple[Any, ...]]]) -> tuple[list[str], list[str], Callable[..., Any]]:
            final = seg[-1][0]
            seg_defs = {p[1] for _, p in seg}
            ins = sorted({n for i, _ in seg for n in uses[i]} - seg_defs - values.keys())
            outs = [p[1] for _, p in seg if p[1] in out_refs or last.get(p[1], -1) > final]
            keep = set(outs)
            # values whose last use is this step (dropped then: an eager segment frees them as soon as MLX has
            # evaluated their consumers, instead of holding every intermediate until the segment ends)
            dead = {i: [n for n in uses[i] if last.get(n) == i and n not in keep and n not in values] for i, _ in seg}

            def fn(*xs: mx.array) -> tuple[mx.array, ...]:
                env: dict[str, Any] = dict(values)
                env.update(zip(ins, xs, strict=True))
                del xs
                for i, p in seg:
                    if p[0] == "alias":
                        env[p[1]] = env[p[2]]
                        if report is not None:
                            report.note(p[1], [env[p[2]]], {}, [p[2]], env)
                    else:
                        _, name, f, args, kwargs = p
                        if f is operator.getitem:
                            env[name] = env[args[0].name][args[1]]
                            if report is not None:
                                report.note(name, [env[name]], {}, [args[0].name], env)
                        elif PROFILE is not None:
                            _profiled(env, name, f, args, kwargs, get)
                        else:
                            env[name] = f(*[get(env, a) for a in args], **{k: get(env, v) for k, v in kwargs.items()})
                            if report is not None:  # `FF_MLX_DTYPE_REPORT=1`
                                report.note(
                                    name,
                                    [get(env, a) for a in args],
                                    {k: get(env, v) for k, v in kwargs.items()},
                                    [n for a in (*args, *kwargs.values()) for n in _refs(a)],
                                    env,
                                )
                    for n in dead[i]:
                        env.pop(n, None)
                return tuple(env[n] for n in outs)

            if (
                not compiled
                or max((self._sizes.get(p[1], 0) for _, p in seg), default=0) > seg_max
                or any(getattr(p[2], "uncompilable", False) for _, p in seg)
            ):
                return ins, outs, fn
            jit = mx.compile(fn)
            kinds[k] = "compiled"
            use: list[Callable[..., Any]] = [jit]
            seg_big = max((self._sizes.get(p[1], 0) for _, p in seg), default=0)
            tune = [seg_big > TUNE_MIN_MB << 20 and not os.environ.get("FF_MLX_NO_TUNE")]
            shape_key = (
                tuple(p[0] if p[0] == "alias" else _op_name(p[2]) for _, p in seg),
                seg_big.bit_length(),
            )
            if known[k] is not None:  # decided by an earlier process (plan cache): no trials
                use[0] = jit if known[k] else fn
                tune[0] = False
                kinds[k] = "known"

            def timed(g: Callable[..., Any], xs: tuple[mx.array, ...]) -> tuple[float, int, Any]:
                """Seconds and peak bytes of the second call of `g` (the first warms kernels and traces)."""
                mx.eval(g(*xs))
                mx.clear_cache()  # the buffer cache can exceed its limit by a big map: one trial's must not stay
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                res = g(*xs)
                mx.eval(res)
                return time.perf_counter() - t0, mx.get_peak_memory(), res

            # the trial spread over the segment's first four calls: compiled, eager (both warm their kernels), then
            # compiled and eager timed; no call replays the segment, so the first call pays no trial
            spread: list[Any] | None = None if os.environ.get("FF_MLX_TUNE_FIRST") else []

            def decide(te: float, me: int, tc: float, mc: int) -> None:
                if tc > te * TUNE_SLACK or mc > me * 1.25 + (64 << 20):
                    use[0] = fn
                decided[k] = use[0] is jit
                self._tune_memo.setdefault(shape_key, decided[k])
                kinds[k] = "tuned-" + ("compiled" if decided[k] else "eager")
                if TUNE_LOG:
                    print(  # noqa: T201 - FF_MLX_TUNE_LOG=1: the tuning pass's trials
                        f"mlx tune: segment {k} ({len(seg)} ops, {seg_big >> 20} MB) eager {1e3 * te:.1f} ms"
                        f" {me >> 20} MB, compiled {1e3 * tc:.1f} ms {mc >> 20} MB"
                        f" -> {'compiled' if decided[k] else 'eager'}",
                        flush=True,
                    )

            def step(xs: tuple[mx.array, ...]) -> tuple[mx.array, ...]:
                """One call of the spread trial (`spread` holds the timings so far)."""
                n = len(spread)
                g = jit if n % 2 == 0 else fn
                if n < 2:
                    spread.append(None)
                    res = g(*xs)
                    if n == 0:
                        kinds[k] = "trial"
                    return res
                mx.eval(xs)  # a pipelined segment before this one has finished
                mx.synchronize()
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                res = g(*xs)
                mx.eval(res)
                spread.append((time.perf_counter() - t0, mx.get_peak_memory()))
                if n == 3:
                    tune[0] = False
                    (tc, mc), (te, me) = spread[2], spread[3]
                    decide(te, me, tc, mc)
                return res

            def guarded(*xs: mx.array) -> tuple[mx.array, ...]:
                if tune[0] and (memo := self._tune_memo.get(shape_key)) is not None:
                    tune[0] = False  # an earlier trial of the same structure decided (recorded with the plan)
                    use[0] = jit if memo else fn
                    decided[k] = memo
                    kinds[k] = "memo-" + ("compiled" if memo else "eager")
                if tune[0] and spread is not None:
                    try:
                        return step(xs)
                    except ValueError as err:
                        if "uncaptured" not in str(err):
                            raise
                        tune[0] = False
                        use[0] = fn
                        decided[k] = False
                        return fn(*xs)
                if tune[0]:  # first call of a segment with big maps: keep the compiled form only if it is not slower
                    tune[0] = False
                    try:
                        te, me, res = timed(fn, xs)
                        del res  # the trials' outputs and freed maps must not add up
                        tc, mc, res = timed(jit, xs)
                    except ValueError as err:
                        if "uncaptured" not in str(err):
                            raise
                        use[0] = fn
                        decided[k] = False
                        return fn(*xs)
                    decide(te, me, tc, mc)
                    return res
                try:
                    return use[0](*xs)
                except ValueError as err:  # a segment mx.compile cannot trace (in-place state): run it eagerly
                    if use[0] is fn or "uncaptured" not in str(err):
                        raise
                    use[0] = fn
                    decided[k] = False
                    return fn(*xs)

            return ins, outs, guarded

        segments = [make(k, steps[a:b]) for k, (a, b) in enumerate(bounds)]
        seg_last = [steps[b - 1][0] for _, b in bounds]
        input_names = [p[1] for p in plan if p[0] == "input"]

        def run(*inputs: mx.array) -> tuple[mx.array, ...]:
            env: dict[str, Any] = {
                n: x.astype(self.dtype) if mx.issubdtype(x.dtype, mx.floating) else x
                for n, x in zip(input_names, inputs, strict=True)
            }
            prev = None
            for k, ((ins, outs, f), end) in enumerate(zip(segments, seg_last, strict=True)):
                xs = [env[n] for n in ins]
                for n in ins:  # inputs that die in this segment: only the segment holds them (freed while it runs)
                    if last.get(n, -1) <= end and n not in out_refs:
                        del env[n]
                res = f(*xs)
                del xs
                env.update(zip(outs, res, strict=True))
                if k + 1 < len(segments):
                    if self.pipeline:
                        mx.async_eval(res)
                        if prev is not None:
                            mx.eval(prev)
                        prev = res
                    else:
                        mx.eval(res)
                for n in [n for n, i in last.items() if i <= end and n in env and n not in out_refs]:
                    del env[n]
            env.update(values)
            if report is not None:
                report.emit(" " + ",".join(str(tuple(x.shape)) for x in inputs))
            if self.plan is not None and decided != known:  # new decisions: the next process skips their trials
                self.plan.save_tune(tune_sig, decided)
                known[:] = decided
            return tuple(o.astype(mx.float32) for o in get(env, out_names))

        if not COLD_LOG:
            return run

        def clocked(f: Callable[..., tuple[mx.array, ...]]) -> Callable[..., tuple[mx.array, ...]]:
            def g(*xs: mx.array) -> tuple[mx.array, ...]:
                if len(cold_runs) > 5 or not cold_runs:
                    return f(*xs)
                t0 = time.perf_counter()
                res = f(*xs)
                mx.eval(res)
                cold_runs[-1].append(time.perf_counter() - t0)
                return res

            return g

        segments[:] = [(ins, outs, clocked(f)) for ins, outs, f in segments]

        def cold_run(*inputs: mx.array) -> tuple[mx.array, ...]:
            if len(cold_runs) < 5:
                cold_runs.append([])
            res = run(*inputs)
            if len(cold_runs) == 5:
                self._cold_report(kinds, [cold_runs[0], cold_runs[4]], bounds)
                cold_runs.append([])  # done: later calls run unclocked
            return res

        return cold_run

    def _cold_report(self, kinds: list[str], runs: list[list[float]], bounds: list[tuple[int, int]]) -> None:
        """`COLD_LOG`: the first call's extra time (first minus fifth call) by phase and segment kind."""
        by: dict[str, list[float]] = {}
        for kind, a, b, (lo, hi) in zip(kinds, *runs, bounds, strict=True):
            rec = by.setdefault(kind, [0, 0.0, 0.0, 0])
            rec[0] += 1
            rec[1] += a - b
            rec[2] += b
            rec[3] += hi - lo
        export = self._cold.get("export", 0.0)
        parts = [
            f"export {export:.2f}s (decomp {self._cold.get('decomp', 0.0):.2f}s)",
            f"translate {self._cold.get('build', 0.0) - export:.2f}s",
        ]
        parts += [f"{k} x{n} ({ops} ops) +{extra:.2f}s (warm {warm:.2f}s)" for k, (n, extra, warm, ops) in by.items()]
        parts.append(f"first {sum(runs[0]):.2f}s fifth {sum(runs[1]):.2f}s")
        print("mlx cold: " + " | ".join(parts), flush=True)  # noqa: T201 - FF_MLX_COLD_LOG=1

    def __call__(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run on CPU tensors; returns float32 CPU tensors."""
        global _KERNELS, _WINO, _HALF_WINO_F32
        _KERNELS = self.kernels
        _HALF_WINO_F32 = self.half_wino_f32
        _WINO = self._wino
        key = tuple((tuple(t.shape), str(t.dtype)) for t in tensors)
        if key not in self._runs:
            self._runs[key] = self._build(tuple(tensors))
            if self.on_built is not None:
                self.on_built()
        outs = self._runs[key](*[to_mx(t) for t in tensors])
        return tuple(to_torch(o) for o in outs)


class _Box:
    """A value cell that `copy.deepcopy` (which `torch.export` applies to a module) shares instead of copying."""

    def __init__(self) -> None:
        self.value: tuple[Any, list[Any], list[Any]] | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> _Box:
        return self


class _Flat(torch.nn.Module):
    """`module(*args, **kwargs)` as a function of its tensor arguments (the rest baked in), flat tensor outputs."""

    def __init__(self, module: torch.nn.Module, spec: Any, leaves: list[Any]) -> None:
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

    def __init__(
        self,
        module: torch.nn.Module,
        dtype: str = "float32",
        *,
        kernels: bool = False,
        release: bool = False,
        compile: bool = True,
        chunk_mb: int = 0,
        pipeline: bool = False,
        half_wino_f32: bool | tuple[tuple[int, int], ...] = False,
    ) -> None:
        """Wrap `module` (eval mode, CPU). `chunk_mb`, `pipeline`, `half_wino_f32`: see `GraphNet`.

        `release=True`: once the first export has moved a parameter or buffer to MLX, the module's own tensor is
        replaced by a one-element stand-in of the same shape (a float32 twin of a 340M-parameter network is 1.4 GB).
        Another input shape exports the same module again and reads its values from MLX, but only the tensors the
        first export used are replaced, and only if no other `MlxModule` shares them. Use it when nothing else runs
        the module's own (CPU) weights.
        """
        self.module = module.eval()
        self.dtype = dtype
        self.kernels = kernels
        self.release = release
        self.compile = compile
        self.chunk_mb = chunk_mb
        self.pipeline = pipeline
        self.half_wino_f32 = half_wino_f32
        self._nets: dict[Any, tuple[GraphNet, _Flat]] = {}
        self._plan_scope = mlx_plan_cache.next_scope()  # this module's key in the on-disk plan cache (a model load's)
        if release:
            for _, t in _tensors(self.module):
                t.__dict__.setdefault("_mlx_owners", set()).add(id(self))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the module on MLX."""
        leaves, spec = pytree.tree_flatten((args, kwargs))
        key = (spec, tuple((tuple(x.shape), x.dtype) if isinstance(x, torch.Tensor) else repr(x) for x in leaves))
        if key not in self._nets:
            flat = _Flat(self.module, spec, leaves)
            self._nets[key] = (
                GraphNet(
                    flat,
                    self.dtype,
                    compile=self.compile,
                    kernels=self.kernels,
                    shared=self._consts(),
                    chunk_mb=self.chunk_mb,
                    pipeline=self.pipeline,
                    half_wino_f32=self.half_wino_f32,
                ),
                flat,
            )
            net = self._nets[key][0]
            sig = (repr(key), self.dtype, self.kernels, self.compile, self.chunk_mb, self.half_wino_f32)
            net.plan = mlx_plan_cache.Entry.make(self._plan_scope, flat, sig)
            net.plan_extra = lambda flat=flat: flat.out.value
            net.plan_restore = lambda extra, flat=flat: setattr(flat.out, "value", extra)
        net, flat = self._nets[key]
        net.on_built = (lambda: self._release(flat)) if self.release else None
        outs = iter(net(*[x for x in leaves if isinstance(x, torch.Tensor)]))
        assert flat.out.value is not None  # noqa: S101 - narrows the Optional; set while exporting
        out_spec, others, dtypes = flat.out.value
        merged = [next(outs).to(d) if d is not None else o for o, d in zip(others, dtypes, strict=True)]
        return pytree.tree_unflatten(merged, out_spec)

    def _release(self, flat: torch.nn.Module) -> None:
        """Replace the twin's tensors that MLX holds (and that no other `MlxModule` still needs) by stand-ins."""
        consumed = self._consts().get(_CONSUMED, set())
        for name, host, leaf, t in _tensors(flat, names=True):
            if name not in consumed:
                continue
            done = t.__dict__.setdefault("_mlx_done", set())
            done.add(id(self))
            if not t.__dict__.get("_mlx_owners", set()) <= done or t.numel() <= 1:
                continue
            stand_in = torch.zeros((), dtype=t.dtype).expand(t.shape)
            old = t.untyped_storage()
            hand_back = old.nbytes() >= _HAND_BACK_MIN and _exclusive(old) and not getattr(old, "filename", None)
            if leaf in host._parameters:
                t.data = stand_in
            else:
                host._buffers[leaf] = stand_in
            if hand_back:
                _hand_back(old)
        gc.collect()

    def parameters(self) -> Any:
        """The wrapped module's (CPU float32) parameters, for runners that read the dtype from them."""
        return self.module.parameters()

    def _consts(self) -> dict[str, Any]:
        return self.__dict__.setdefault("_shared_consts", {})


def _tensors(module: torch.nn.Module, *, names: bool = False) -> Iterator[Any]:
    """The module's parameters and buffers: `(name, host, leaf name, tensor)` if `names`, else `(None, tensor)`."""
    seen: set[int] = set()
    for prefix, host in module.named_modules(remove_duplicate=False):
        for leaf, t in (*host._parameters.items(), *host._buffers.items()):
            if t is None or (id(t) in seen and not names):
                continue
            seen.add(id(t))
            yield (f"{prefix}.{leaf}" if prefix else leaf, host, leaf, t) if names else (None, t)


class _Name:
    """A graph value by name (a folded constant)."""

    def __init__(self, name: str) -> None:
        self.name = name
