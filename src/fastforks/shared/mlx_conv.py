"""NHWC convolutions for the hand-written MLX CNNs (edge detectors, restoration nets): one `Conv` per `nn.Conv2d`.

A `Conv` keeps its weights as arrays in the run dtype (OHWI, read through `mlx_port.tensor`, so a cache hit loads
them without the checkpoint) and applies them as the graph translation does (`mlx_graph`): 1x1 convolutions as one
GEMM where `conv1x1_ok` says so, depthwise convolutions in `graph_mlx`'s kernel, the rest through `mlx_graph._conv2d`
(ragged channel counts padded to MLX's GEMM tiles), the bias added afterwards. The translator's dispatch reads
module-level flags; `mode` sets them around a call (a compiled step reads them while it is traced).
"""

# ruff: noqa: ANN401 - torch modules

from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.shared import mlx_graph as mg
from fastforks.shared.swin_sr import mlx_net as swin

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["Conv", "ConvT", "instance_norm", "mode", "reflect_pad", "resize"]


@contextlib.contextmanager
def mode(*, kernels: bool = False, half_wino_f32: bool = False) -> Generator[None]:
    """The translator's convolution flags while a network is traced or run (restored afterwards).

    `half_wino_f32`: in float16, 3x3 convolutions MLX would run as Winograd (error ~17x the direct conv's) run in
    float32 instead.
    """
    saved = mg._KERNELS, mg._GEMM1X1, mg._HALF_WINO_F32, mg._HALF_WINO_PAD  # noqa: SLF001
    mg._KERNELS, mg._GEMM1X1, mg._HALF_WINO_F32, mg._HALF_WINO_PAD = kernels, True, half_wino_f32, False  # noqa: SLF001
    try:
        yield
    finally:
        mg._KERNELS, mg._GEMM1X1, mg._HALF_WINO_F32, mg._HALF_WINO_PAD = saved  # noqa: SLF001


def _pair(v: Any) -> tuple[int, int]:
    return (int(v[0]), int(v[1]))


class Conv:
    """An `nn.Conv2d` on NHWC maps in `dtype` (`scale`: the weights and bias times it, folded in the conversion)."""

    def __init__(  # noqa: PLR0913
        self,
        layer: Any,
        dtype: mx.Dtype,
        *,
        scale: float = 1.0,
        bn: Any = None,
        out: slice | None = None,
        bias: bool = True,
        more: tuple[Any, ...] = (),
        shift: bool = True,
    ) -> None:
        """The arrays of `layer` (float32, or the meta skeleton on a cache hit).

        `bn`: the `BatchNorm2d` after it (eval mode), folded into the weights and bias in the conversion (`shift=False`:
        only its scale, the shift belongs to another convolution whose output is added). `out`: keep only these output
        channels. `bias=False` leaves the bias out (a norm follows that subtracts it again). `more`: bias-free layers
        of the same geometry applied to further input channels (concatenated after `layer`'s) whose outputs add to its.
        """
        w, b = layer.weight, layer.bias
        if bn is None and out is None and not more:
            self.w = swin.arr(w.permute(0, 2, 3, 1), dtype, scale=scale)  # OHWI
            self.b = None if b is None or not bias else swin.arr(b, dtype, scale=scale)
        else:
            sel = out or slice(None)
            co = len(range(*sel.indices(w.shape[0])))
            ci = w.shape[1] + sum(m.weight.shape[1] for m in more)

            def fold() -> tuple[np.ndarray, np.ndarray]:
                wd = np.concatenate([m.weight.detach().double().numpy()[sel] for m in (layer, *more)], axis=1)
                bd = np.zeros(wd.shape[0]) if b is None else b.detach().double().numpy()[sel]
                if bn is not None:
                    k = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy() + bn.eps)
                    k, mean, beta = (
                        k[sel],
                        bn.running_mean.detach().double().numpy()[sel],
                        bn.bias.detach().double().numpy()[sel],
                    )
                    wd, bd = wd * k[:, None, None, None], bd * k + ((beta - mean * k) if shift else 0)
                return wd.transpose(0, 2, 3, 1) * scale, bd * scale

            shape = (co, w.shape[2], w.shape[3], ci)
            self.w = swin.derived(shape, dtype, lambda: fold()[0])
            self.b = swin.derived((co,), dtype, lambda: fold()[1]) if bias else None
        self.stride, self.padding = _pair(layer.stride), _pad(layer.padding)
        self.dilation, self.groups = _pair(layer.dilation), int(layer.groups)
        co, kh, kw, ci = self.w.shape
        self.depthwise = self.groups > 1 and ci == 1 and co == self.groups and kh * kw > 1
        self.gemm = self.groups == 1 and kh * kw == 1 and self.stride == (1, 1) and self.padding == (0, 0)
        self.wm = self.w.reshape(co, ci).T if self.gemm else None  # [in, out], a view of `w`

    def __call__(self, x: mx.array) -> mx.array:
        """The convolution (and its bias) of the NHWC map `x`."""
        co, _, _, ci = self.w.shape
        if self.wm is not None and mg.conv1x1_ok(
            ci, co, x.shape[0] * x.shape[1] * x.shape[2], (1, 1), (1, 1), (0, 0), 1
        ):
            return mg.conv1x1(x, self.wm, self.b)
        if self.depthwise and co % 4 == 0 and mg.DW_KERNEL:
            from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415 - Metal only

            bias = self.b if self.b is not None else mx.zeros((co,), dtype=x.dtype)
            return ops.depthwise(
                x, self.w[:, :, :, 0].transpose(1, 2, 0), bias, self.stride, self.padding, self.dilation
            )
        y = mg._conv2d(x, self.w, self.stride, self.padding, self.dilation, self.groups)  # noqa: SLF001
        return y if self.b is None else y + self.b


def _pad(p: Any) -> tuple[int, int]:
    if isinstance(p, str):
        msg = f"MLX conv: numeric padding expected, got {p!r}"
        raise NotImplementedError(msg)
    return _pair(p)


@functools.cache
def _taps(n_in: int, n_out: int, *, align_corners: bool) -> np.ndarray:
    """The `[n_out, n_in]` matrix of torch's bilinear resize along one axis."""
    if align_corners:
        src = np.arange(n_out, dtype=np.float64) * ((n_in - 1) / (n_out - 1) if n_out > 1 else 0.0)
    else:
        src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * n_in / n_out - 0.5, 0.0)
    i0 = np.minimum(src.astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = src - i0
    m = np.zeros((n_out, n_in), np.float32)
    rows = np.arange(n_out)
    np.add.at(m, (rows, i0), (1 - w1).astype(np.float32))
    np.add.at(m, (rows, i1), w1.astype(np.float32))
    return m


def resize(x: mx.array, size: tuple[int, int], *, align_corners: bool = False) -> mx.array:
    """`F.interpolate(mode="bilinear")` of the NHWC map `x` to `size`, as one matrix product per axis."""
    n, h, w, c = x.shape
    if w != size[1]:
        m = mx.array(_taps(w, size[1], align_corners=align_corners)).astype(x.dtype)
        x = mx.matmul(m, x.reshape(n * h, w, c)).reshape(n, h, size[1], c)
    if h != size[0]:
        m = mx.array(_taps(h, size[0], align_corners=align_corners)).astype(x.dtype)
        x = mx.matmul(m, x.reshape(n, h, size[1] * c)).reshape(n, size[0], size[1], c)
    return x


def _up_taps(k: int, r: int) -> list[tuple[int, int]]:
    """(input offset, kernel tap) pairs of output phase `r` of a stride-2, padding-1 transposed convolution."""
    return [(off, r + 1 - 2 * off) for off in range(-2, 3) if 0 <= r + 1 - 2 * off < k]


class ConvT:
    """An `nn.ConvTranspose2d` (stride 2, padding 1, output twice the input) on NHWC maps.

    One small convolution per output phase (`mx.conv_transpose2d` convolves the zero-stuffed map: 4x the taps).

    `bias=False` leaves the bias out (a norm follows that subtracts it again); `bn`: the `BatchNorm2d` after it, folded.
    """

    def __init__(self, layer: Any, dtype: mx.Dtype, *, bias: bool = True, bn: Any = None) -> None:
        """The arrays of `layer` (float32, or the meta skeleton on a cache hit)."""
        w = layer.weight  # [Cin, Cout, k, k]
        k = int(w.shape[2])
        if _pair(layer.stride) != (2, 2) or _pad(layer.padding) != (1, 1) or k not in (3, 4):
            msg = "MLX ConvT: only stride 2, padding 1, kernel 3 or 4"
            raise NotImplementedError(msg)
        ci, co = w.shape[:2]
        self.phases: list[
            tuple[int, int, int, int, mx.array]
        ] = []  # (row offset, column offset, rows, columns, weights)

        def scale() -> np.ndarray:
            if bn is None:
                return np.ones(co)
            return bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy() + bn.eps)

        for rh in range(2):
            for rw in range(2):
                th, tw = _up_taps(k, rh), _up_taps(k, rw)

                def make(th: list = th, tw: list = tw) -> np.ndarray:
                    wd = w.detach().double().numpy() * scale()[None, :, None, None]
                    return np.stack([np.stack([wd[:, :, a[1], b[1]].T for b in tw], axis=1) for a in th], axis=1)

                weights = swin.derived((co, len(th), len(tw), ci), dtype, make)
                self.phases.append((th[0][0], tw[0][0], len(th), len(tw), weights))

        def shift() -> np.ndarray:
            bd = np.zeros(co) if layer.bias is None or not bias else layer.bias.detach().double().numpy()
            if bn is None:
                return bd
            return (
                bd * scale() + bn.bias.detach().double().numpy() - bn.running_mean.detach().double().numpy() * scale()
            )

        self.b = swin.derived((co,), dtype, shift) if (bias and layer.bias is not None) or bn is not None else None

    def __call__(self, x: mx.array) -> mx.array:
        """The transposed convolution of the NHWC map `x`."""
        n, h, w, _ = x.shape
        xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        ys = [
            mx.conv2d(xp[:, 1 + oh : 1 + oh + h + kh - 1, 1 + ow : 1 + ow + w + kw - 1], wt)
            for oh, ow, kh, kw, wt in self.phases
        ]
        y = mx.stack(ys, axis=3).reshape(n, h, w, 2, 2, -1).transpose(0, 1, 3, 2, 4, 5).reshape(n, 2 * h, 2 * w, -1)
        return y if self.b is None else y + self.b


def instance_norm(x: mx.array, *, relu: bool = False, eps: float = 1e-5) -> mx.array:
    """`nn.InstanceNorm2d` (no affine) of an NHWC map with float32 statistics (one pass), the ReLU optionally fused."""
    from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415 - Metal only

    n, h, w, c = x.shape
    stats = [ops.channel_stats(x[i].reshape(h * w, c)) for i in range(n)]
    mean = mx.stack([s[0] for s in stats]).reshape(n, 1, 1, c)
    scale = mx.rsqrt(mx.stack([s[1] for s in stats]).reshape(n, 1, 1, c) + eps)
    y = (x.astype(mx.float32) - mean) * scale
    return (mx.maximum(y, 0) if relu else y).astype(x.dtype)


def reflect_pad(x: mx.array, p: int) -> mx.array:
    """`F.pad(x, (p, p, p, p), mode="reflect")` of an NHWC map (two gathers)."""
    if p == 0:
        return x
    for axis in (1, 2):
        n = x.shape[axis]
        idx = np.concatenate([np.arange(p, 0, -1), np.arange(n), np.arange(n - 2, n - 2 - p, -1)])
        x = mx.take(x, mx.array(idx.astype(np.uint32)), axis=axis)
    return x
