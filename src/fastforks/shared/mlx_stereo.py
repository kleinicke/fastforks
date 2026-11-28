"""Building blocks of the hand-written MLX stereo networks: weights through the conversion cache, folded convolutions.

For the stereo families' `mlx_net.py` (fast_acvnet, gwcnet, ...). A builder walks the vendored PyTorch network and
makes every weight-derived array through `Weights`, which declares the array's final shape up front and reads it via
`mlx_port.tensor`: the first load folds the batch norms and transposes in numpy (`Weights.conv` ...) and records the
arrays, every later load takes them from the conversion cache without a checkpoint (the module `mlx_port`).

Layouts (channels last): a 2-D convolution weight `[O, kh, kw, I]`, a 3-D one `[O, kd, kh, kw, I]`, a transposed 2-D
one `[O, kh, kw, I]`, a transposed 3-D one stays `[I, O, kd, kh, kw]` (the layout `mlx_graph._conv_transpose3d_phases`
takes), a depthwise one `[kh, kw, C]`.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import depthwise
from fastforks.shared import mlx_port
from fastforks.shared.mlx_graph import _conv_transpose3d_phases

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Weights",
    "arr",
    "concat_volume",
    "conv2d",
    "conv3d",
    "deconv3d",
    "folded",
    "leaky",
    "mobile",
    "relu",
    "relu6",
    "resnet",
]

Conv = tuple[mx.array, mx.array, int, int]  # weight [O, k.., I], bias, stride, padding
Dilated = tuple[mx.array, mx.array, int, int, int]  # the same plus the dilation


def arr(shape: tuple[int, ...], dtype: mx.Dtype, make: Callable[[], np.ndarray]) -> mx.array:
    """The array of `shape` that `make()` computes from weight values, in `dtype` (cached: `make` not run on a hit)."""

    def convert() -> mx.array:
        a = make()
        if a.shape != shape:
            msg = f"MLX stereo port: built {a.shape}, declared {shape}"
            raise AssertionError(msg)
        return mx.array(np.ascontiguousarray(a, np.float32)).astype(dtype)

    return mlx_port.tensor(SimpleNamespace(shape=shape), dtype, convert)


def folded(conv: Any, norm: Any = None, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch modules
    """A convolution's weight and bias with its batch norm (or None) folded in; `axis`: the weight's output axis."""
    w = conv.weight.detach().float().numpy()
    b = np.zeros(w.shape[axis], np.float32) if conv.bias is None else conv.bias.detach().float().numpy()
    if norm is not None:
        scale = norm.weight.detach().float().numpy().astype(np.float64) / np.sqrt(
            norm.running_var.float().numpy().astype(np.float64) + norm.eps
        )
        shape = [1] * w.ndim
        shape[axis] = -1
        w = (w * scale.reshape(shape)).astype(np.float32)
        b = (norm.bias.detach().float().numpy() + (b - norm.running_mean.float().numpy()) * scale).astype(np.float32)
    return w, b


def leaky(x: mx.array) -> mx.array:
    """LeakyReLU(0.01)."""
    return mx.maximum(x, 0.01 * x)


def relu(x: mx.array) -> mx.array:
    """ReLU."""
    return mx.maximum(x, 0)


def relu6(x: mx.array) -> mx.array:
    """ReLU6."""
    return mx.clip(x, 0, 6)


def _norm(m: Any) -> Any:  # noqa: ANN401 - a vendored `BasicConv`
    return m.bn if m.use_bn else None


def _cb(seq: Any) -> Any:  # noqa: ANN401 - torch modules
    """The `(conv, batch norm)` pair of a `convbn(...)` Sequential, or of the Sequential `(convbn, ReLU)` around it."""
    return seq if hasattr(seq[0], "weight") else seq[0]


class Weights:
    """Cache-ready copies of the vendored modules' weights in one MLX dtype."""

    def __init__(self, dtype: mx.Dtype) -> None:
        """Weights read in `dtype`."""
        self.dtype = dtype

    def a(self, shape: tuple[int, ...], make: Callable[[], np.ndarray]) -> mx.array:
        """`arr` in this dtype."""
        return arr(shape, self.dtype, make)

    def _pair(
        self,
        conv: Any,  # noqa: ANN401 - torch modules
        norm: Any,  # noqa: ANN401
        perm: tuple[int, ...],
        axis: int = 0,
    ) -> tuple[mx.array, mx.array]:
        fold = functools.cache(lambda: folded(conv, norm, axis))
        shape = tuple(int(n) for n in conv.weight.shape)
        out = shape[axis]
        return (
            self.a(tuple(shape[i] for i in perm), lambda: fold()[0].transpose(perm)),
            self.a((out,), lambda: fold()[1]),
        )

    def conv(self, conv: Any, norm: Any = None) -> Conv:  # noqa: ANN401 - torch modules of the vendored net
        """A 2-D / 3-D convolution (batch norm folded in): weight `[O, k.., I]`, bias, stride, padding."""
        nd = conv.weight.ndim
        perm = (0, *range(2, nd), 1)
        w, b = self._pair(conv, norm, perm)
        return w, b, int(conv.stride[0]), int(conv.padding[0])

    def basic(self, m: Any) -> Conv:  # noqa: ANN401 - a vendored `BasicConv` (conv, bn, relu flags)
        """`BasicConv` (2-D or 3-D)."""
        return self.conv(m.conv, _norm(m))

    def deconv(self, conv: Any, norm: Any = None) -> Conv:  # noqa: ANN401
        """A transposed 2-D convolution: weight `[Cout, kh, kw, Cin]`."""
        w, b = self._pair(conv, norm, (1, 2, 3, 0), axis=1)
        return w, b, int(conv.stride[0]), int(conv.padding[0])

    def deconv3(self, m: Any) -> tuple[mx.array, mx.array, bool]:  # noqa: ANN401
        """A transposed 3-D `BasicConv`: weight `[Cin, Cout, kd, kh, kw]`, bias, activation flag."""
        w, b = self._pair(m.conv, _norm(m), (0, 1, 2, 3, 4), axis=1)
        return w, b, m.relu

    def cb(self, seq: Any) -> Conv:  # noqa: ANN401
        """A `convbn(...)` Sequential (optionally followed by its ReLU in a Sequential): batch norm folded."""
        pair = _cb(seq)
        return self.conv(pair[0], pair[1])

    def dilated(self, seq: Any) -> Dilated:  # noqa: ANN401
        """`cb` of a 2-D `convbn` plus its dilation."""
        pair = _cb(seq)
        return (*self.conv(pair[0], pair[1]), int(pair[0].dilation[0]))

    def residual(self, b: Any) -> tuple[Dilated, Dilated, Dilated | None]:  # noqa: ANN401
        """A BasicBlock (conv1 + ReLU, conv2, optional 1 x 1 downsample); no ReLU after the add."""
        down = None if b.downsample is None else (*self.conv(b.downsample[0], b.downsample[1]), 1)
        return self.dilated(b.conv1), self.dilated(b.conv2), down

    def hourglass3(self, h: Any) -> dict[str, Any]:  # noqa: ANN401
        """The classic 3-D hourglass (PSMNet / GwcNet): conv1-4 `cb`, conv5 / 6 transposed, `redir*` if present."""
        out: dict[str, Any] = {k: self.cb(getattr(h, k)) for k in ("conv1", "conv2", "conv3", "conv4")}
        for k in ("conv5", "conv6"):
            seq = getattr(h, k)
            out[k] = self.deconv3(SimpleNamespace(conv=seq[0], bn=seq[1], use_bn=True, relu=False))
        for k in ("redir1", "redir2"):
            if hasattr(h, k):
                out[k] = self.cb(getattr(h, k))
        return out

    def conv2x(self, m: Any) -> tuple[Conv, Conv]:  # noqa: ANN401
        """`Conv2x` with a transposed convolution: (deconv, merge convolution)."""
        return self.deconv(m.conv1.conv, m.conv1.bn), self.basic(m.conv2)

    def stem(self, seq: Any) -> tuple[Conv, Conv]:  # noqa: ANN401
        """`BasicConv -> Conv2d -> BatchNorm2d -> ReLU`."""
        return self.basic(seq[0]), self.conv(seq[1], seq[2])

    def att(self, m: Any) -> tuple[Conv, Conv]:  # noqa: ANN401
        """`channelAtt`'s image branch."""
        return self.basic(m.im_att[0]), self.conv(m.im_att[1])

    def dw(self, conv: Any, norm: Any) -> tuple[mx.array, mx.array, int]:  # noqa: ANN401
        """A depthwise convolution: weight `[kh, kw, C]`, bias, stride."""
        fold = functools.cache(lambda: folded(conv, norm))
        c, _, kh, kw = (int(n) for n in conv.weight.shape)
        return (
            self.a((kh, kw, c), lambda: fold()[0][:, 0].transpose(1, 2, 0)),
            self.a((c,), lambda: fold()[1]),
            int(conv.stride[0]),
        )

    def block(self, m: Any) -> tuple[Any, ...]:  # noqa: ANN401
        """A MobileNetV2 block: (expand or None, depthwise, project, skip)."""
        if hasattr(m, "conv_pwl"):
            return self.conv(m.conv_pw, m.bn1), self.dw(m.conv_dw, m.bn2), self.conv(m.conv_pwl, m.bn3), m.has_skip
        return None, self.dw(m.conv_dw, m.bn1), self.conv(m.conv_pw, m.bn2), m.has_skip


def mobile(x: mx.array, blocks: list[tuple[Any, ...]], conv: Callable[[mx.array, Conv], mx.array]) -> mx.array:
    """A run of MobileNetV2 blocks (`Weights.block`); `conv(x, c)` applies a 1 x 1 convolution."""
    for expand, (w, b, stride), project, skip in blocks:
        y = x if expand is None else relu6(conv(x, expand))
        y = relu6(depthwise(y, w, b, (stride, stride), (1, 1), (1, 1)))
        y = conv(y, project)
        x = x + y if skip else y
    return x


def conv2d(x: mx.array, c: Dilated) -> mx.array:
    """A 2-D convolution of `Weights.dilated` (NHWC)."""
    w, b, stride, padding, dilation = c
    return mx.conv2d(x, w, stride=stride, padding=padding, dilation=dilation) + b


def conv3d(x: mx.array, c: Conv, *, act: bool = True) -> mx.array:
    """A 3-D convolution of `Weights.conv` (NDHWC), ReLU after it unless `act` is False."""
    w, b, stride, padding = c
    y = mx.conv3d(x, w, stride=stride, padding=padding) + b
    return relu(y) if act else y


def deconv3d(x: mx.array, up: tuple[mx.array, mx.array, bool]) -> mx.array:
    """A stride-2 transposed 3-D convolution (`Weights.deconv3`; kernel 3, padding 1, output padding 1)."""
    w, b, _ = up
    y = _conv_transpose3d_phases(x, w, (2, 2, 2), (1, 1, 1), (1, 1, 1))
    if y is None:  # pragma: no cover - the pair is padded to 32s, every volume level halves exactly
        msg = f"MLX stereo port: the cost volume {tuple(x.shape[1:4])} does not halve evenly"
        raise ValueError(msg)
    return y + b


def resnet(x: mx.array, first: list[Dilated], layers: list[list[Any]]) -> list[mx.array]:
    """The GwcNet / PSMNet feature trunk (`firstconv`, `layer1-4`): the output of each layer, NHWC."""
    for c in first:
        x = relu(conv2d(x, c))
    outs = []
    for blocks in layers:
        for c1, c2, down in blocks:
            y = conv2d(relu(conv2d(x, c1)), c2)
            x = y + (x if down is None else conv2d(x, down))
        outs.append(x)
    return outs


def concat_volume(left: mx.array, right: mx.array, depth: int) -> mx.array:
    """Upstream's `build_concat_volume`: `[1, D, H, W, 2C]` of `[1, H, W, C]` features (left kept where `x >= d`)."""
    wd = left.shape[2]
    rp = mx.pad(right, [(0, 0), (0, 0), (depth, 0), (0, 0)])
    cols = mx.arange(wd)[None, :, None]
    zero = mx.zeros((), left.dtype)
    slabs = [
        mx.concatenate([mx.where(cols >= i, left[0], zero), rp[0, :, depth - i : depth - i + wd]], axis=-1)
        for i in range(depth)
    ]
    return mx.stack(slabs)[None]
