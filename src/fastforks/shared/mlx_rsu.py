"""Residual U-blocks (U²-Net's `RSU7` ... `RSU4F`, IS-Net's) on Apple MLX, hand-written: NHWC, the run dtype.

The maths of the vendored torch modules (`matting/models/u2net/net`, `matting/models/isnet/net`), read by structure:

- `REBNCONV` (3 x 3 conv, dilated, BatchNorm, ReLU) is one conv with the BatchNorm folded in at load, in float64 in
  the numpy conversion (some unnormalised conv outputs exceed the float16 range: 87 548 in U²-Net's `stage2d`; the
  folded ones stay inside it), dispatched as the graph translator's (`mlx_graph._conv2d`: ragged channel counts padded
  to MLX's GEMM tiles); kernels mode takes `conv_mlx`'s Winograd 3 x 3 for undilated float16 convs;
- `MaxPool2d(2, 2, ceil_mode=True)` as the maximum over a reshaped window (odd sides padded with -inf);
- `_upsample_like` (bilinear, `align_corners=False`) as the NHWC resize kernel (`graph_mlx.ops.bilinear_nhwc`).

An RSU of `L` levels: `rebnconvin`, the encoder convs `rebnconv1 .. rebnconvL` (a pool after level `i` where the
block has `pool{i}`), the decoder convs `rebnconv{L-1}d .. rebnconv1d` on the concatenation of the level above
(upsampled to the level's size after a pool) with the level's encoder output, plus the input conv's output.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared import mlx_port

if TYPE_CHECKING:
    from collections.abc import Callable

    Layer = Callable[[mx.array], mx.array]

__all__ = ["conv", "conv_of", "options", "pool2", "rebn", "rsu", "up_like"]


def _shape(*dims: int) -> Any:
    """A stand-in carrying only the shape `mlx_port.tensor` checks on a replay (no torch op on the meta skeleton)."""
    return SimpleNamespace(shape=tuple(int(d) for d in dims))


def _np(t: Any) -> np.ndarray:
    return t.detach().double().numpy()


def conv(layer: Any, bn: Any | None, dtype: mx.Dtype, *, kernels: bool = False) -> Layer:
    """A `Conv2d` (an eval `BatchNorm2d` after it folded in, if given) on NHWC maps, the bias added.

    The weights are converted in numpy (float64 fold, OHWI); on a cache hit only the shapes of `layer` are read.
    """
    o = int(layer.weight.shape[0])

    def folded() -> tuple[np.ndarray, np.ndarray]:
        w = _np(layer.weight)
        b = np.zeros(o) if layer.bias is None else _np(layer.bias)
        if bn is not None:
            s = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps)
            w, b = w * s[:, None, None, None], _np(bn.bias) + (b - _np(bn.running_mean)) * s
        return w, b

    return conv_of(
        tuple(int(d) for d in layer.weight.shape), folded, dtype, stride=_pair(layer.stride),
        padding=_pair(layer.padding), dilation=_pair(layer.dilation), groups=int(layer.groups), kernels=kernels,
    )  # fmt: skip


def _pair(v: Any) -> tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def conv_of(  # noqa: PLR0913 - a Conv2d's geometry
    oihw: tuple[int, ...],
    folded: Callable[[], tuple[np.ndarray, np.ndarray]],
    dtype: mx.Dtype,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    kernels: bool = False,
) -> Layer:
    """A conv of `[O, I, kh, kw]` weights on NHWC maps; `folded()` makes the OIHW weight and the bias (numpy).

    `folded` runs only when the conversion cache misses (`mlx_port.tensor`).
    """
    o, i, kh, kw = oihw

    def mlx(a: np.ndarray) -> mx.array:
        out = mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype)
        mx.eval(out)
        return out

    w = mlx_port.tensor(_shape(o, kh, kw, i), dtype, lambda: mlx(folded()[0].transpose(0, 2, 3, 1)))
    b = mlx_port.tensor(_shape(o), dtype, lambda: mlx(folded()[1]))
    if (kh, kw) == (1, 1) and stride == (1, 1) and groups == 1:
        return lambda x: mg.conv1x1(x, w.reshape(o, i).T, b)
    spec: Any = SimpleNamespace(shape=(o, i, kh, kw))  # the translator's checks and its Winograd weight cache key

    def apply(x: mx.array) -> mx.array:
        if kernels and groups == 1 and dilation == (1, 1):
            from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - kernels mode only

            if mg._direct_ok(x.shape[3], x.shape[1] * x.shape[2], spec.shape, stride):  # noqa: SLF001
                return conv_ops.direct(x, w, b, stride, padding, dilation)
            if mg._wino_ok(x, spec, stride, padding, dilation, groups):  # noqa: SLF001
                t = mg._wino_tile(x)  # noqa: SLF001
                return conv_ops.conv3x3(x, mg._wino_weights(spec, w, x.dtype, t), b, m=t)  # noqa: SLF001
        return mg._conv2d(x, w, stride, padding, dilation, groups) + b  # noqa: SLF001

    return apply


def options(*, kernels: bool = False) -> None:
    """The translator's module options `conv` reads (`mlx_graph._conv2d`), set for a call: kernels mode's padding."""
    mg._KERNELS, mg._GEMM1X1 = kernels, True  # noqa: SLF001
    mg._HALF_WINO_F32, mg._HALF_WINO_PAD = False, False  # noqa: SLF001


def rebn(m: Any, dtype: mx.Dtype, *, kernels: bool = False) -> Layer:
    """`REBNCONV`: ReLU(BatchNorm(conv(x))), the BatchNorm folded."""
    c = conv(m.conv_s1, m.bn_s1, dtype, kernels=kernels)
    return lambda x: mx.maximum(c(x), 0)


def pool2(x: mx.array) -> mx.array:
    """`MaxPool2d(2, stride=2, ceil_mode=True)` on NHWC."""
    n, h, w, c = x.shape
    if h % 2 or w % 2:
        x = mx.pad(x, [(0, 0), (0, h % 2), (0, w % 2), (0, 0)], constant_values=-float("inf"))
        h, w = h + h % 2, w + w % 2
    return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


TINY_MAP = 8  # MLX passes kernel inputs of fewer elements as `constant` buffers, which the resize kernel cannot read


@functools.lru_cache(maxsize=64)
def _lerp(n: int, out: int, align: bool) -> mx.array:  # noqa: FBT001 - cache key
    """`[out, n]` weights of torch's bilinear resize along one axis."""
    if align:
        src = np.arange(out) * ((n - 1) / (out - 1) if out > 1 else 0.0)
    else:
        src = np.maximum((np.arange(out) + 0.5) * (n / out) - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(int), n - 1)
    i1, lam, rows = np.minimum(i0 + 1, n - 1), src - i0, np.arange(out)
    m = np.zeros((out, n))
    np.add.at(m, (rows, i0), 1.0 - lam)
    np.add.at(m, (rows, i1), lam)
    return mx.array(m.astype(np.float32))


def up_like(x: mx.array, hw: tuple[int, int], *, align: bool = False) -> mx.array:
    """`F.interpolate(x, size=hw, mode="bilinear", align_corners=align)` on NHWC."""
    h, w = x.shape[1:3]
    if (h, w) == tuple(hw):
        return x
    if x.size < TINY_MAP:  # a few pixels (small test inputs): the separable weights, float32
        y = mx.einsum("oh,nhwc,pw->nopc", _lerp(h, hw[0], align), x.astype(mx.float32), _lerp(w, hw[1], align))
        return y.astype(x.dtype)
    if align:
        ratios = ((h - 1) / (hw[0] - 1) if hw[0] > 1 else 0.0, (w - 1) / (hw[1] - 1) if hw[1] > 1 else 0.0)
        return ops.bilinear_nhwc(x, hw, ratios, align=True)
    return ops.bilinear_nhwc(x, hw, (h / hw[0], w / hw[1]), align=False)


def rsu(block: Any, dtype: mx.Dtype, *, kernels: bool = False) -> Layer:
    """An `RSU*` block (any depth, pooled or dilated levels) on NHWC."""
    levels = 1
    while hasattr(block, f"rebnconv{levels + 1}"):
        levels += 1
    first = rebn(block.rebnconvin, dtype, kernels=kernels)
    enc = [rebn(getattr(block, f"rebnconv{i}"), dtype, kernels=kernels) for i in range(1, levels + 1)]
    dec = {i: rebn(getattr(block, f"rebnconv{i}d"), dtype, kernels=kernels) for i in range(levels - 1, 0, -1)}
    pooled = [hasattr(block, f"pool{i}") for i in range(1, levels + 1)]

    def apply(x: mx.array) -> mx.array:
        hxin = first(x)
        feats, h = [], hxin
        for i, op in enumerate(enc):
            h = op(h)
            feats.append(h)
            if pooled[i]:
                h = pool2(h)
        d = feats[-1]
        for i in range(levels - 1, 0, -1):
            d = dec[i](mx.concatenate([d, feats[i - 1]], axis=-1))
            if i > 1 and pooled[i - 2]:
                d = up_like(d, feats[i - 2].shape[1:3])
        return d + hxin

    return apply
