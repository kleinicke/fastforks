"""The IGEV-family blocks on Apple MLX in `mlx.core` operations (NHWC maps, NDHWC volumes, batch norms folded).

Weights are copied from the vendored torch modules (`net/`) by `Weights(dtype)` (cache-ready: `mlx_port.tensor`); the
forward functions mirror the modules of `net/extractor.py` and `net/submodule.py` and the cost-volume `hourglass` the
families share:

- `feature`: the MobileNetV2 trunk with its up-sampling (`Feature`), `[x4, x8, x16, x32]`;
- `context`: the context encoder (`MultiBasicEncoder`, batch norms), per level `(net, inp)`;
- `hourglass`: the 3-D cost aggregation with its image-feature gates (`FeatureAtt`);
- `GeoLookup`: the combined geometry / all-pairs correlation volume and its 1-D lookups (`FastGeoEncodingVolume`);
- `gru`: `ConvGRU` with the z / r convolutions as one, `pool2x`, `interp`.

Instance norms take float32 statistics; everything else runs in the activations' dtype.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc, depthwise, linear_sample
from fastforks.shared import mlx_port
from fastforks.shared.mlx_graph import _conv_transpose3d_phases

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Conv",
    "GeoLookup",
    "Weights",
    "arrays",
    "basic",
    "context",
    "context_upsample",
    "conv",
    "deconv",
    "feature",
    "gate",
    "gru",
    "hourglass",
    "inorm",
    "interp",
    "leaky",
    "pool2x",
    "relu",
]

Conv = tuple[mx.array, mx.array, int | tuple[int, ...], int | tuple[int, ...]]  # weight [O, k.., I], bias, stride, pad
IN_EPS = 1e-05  # nn.InstanceNorm2d's


def _folded(m: Any, norm: Any, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - torch modules
    """A convolution's weight and bias with its batch norm (or None) folded in (float64); `axis`: the output axis."""
    w = m.weight.detach().float().numpy()
    b = np.zeros(w.shape[axis], np.float32) if m.bias is None else m.bias.detach().float().numpy()
    if norm is not None:
        scale = norm.weight.detach().double().numpy() / np.sqrt(norm.running_var.double().numpy() + norm.eps)
        shape = [1] * w.ndim
        shape[axis] = -1
        w = (w * scale.reshape(shape)).astype(np.float32)
        b = (norm.bias.detach().double().numpy() + (b - norm.running_mean.double().numpy()) * scale).astype(np.float32)
    return w, b


def leaky(x: mx.array) -> mx.array:
    """`nn.LeakyReLU()` (slope 0.01)."""
    return mx.maximum(x, 0.01 * x)


def relu(x: mx.array) -> mx.array:
    """`nn.ReLU`."""
    return mx.maximum(x, 0)


def _relu6(x: mx.array) -> mx.array:
    return mx.clip(x, 0, 6)


def inorm(x: mx.array) -> mx.array:
    """`nn.InstanceNorm2d/3d(affine=False)` of a channels-last map: float32 statistics, one rounding."""
    xf = x.astype(mx.float32)
    axes = tuple(range(1, x.ndim - 1))  # every spatial axis (3D volumes: D, H, W)
    mean = xf.mean(axis=axes, keepdims=True)
    var = mx.square(xf - mean).mean(axis=axes, keepdims=True)
    return ((xf - mean) * mx.rsqrt(var + IN_EPS)).astype(x.dtype)


def _ints(v: Any) -> int | tuple[int, ...]:  # noqa: ANN401 - a module's stride / padding
    """One int when every axis agrees (what `mx.conv2d` / `mx.conv3d` and the transposed kernels take), else a tuple."""
    t = tuple(int(n) for n in v) if isinstance(v, (tuple, list)) else (int(v),)
    return t[0] if len(set(t)) == 1 else t


class Weights:
    """Cache-ready copies of the vendored modules' weights in one MLX dtype.

    Every weight-derived array is declared with its final shape and read through `mlx_port.tensor`: built in NumPy
    (batch norms folded, transposed) on a first load, taken from the conversion cache on a `mlx_port.recorded` hit
    (outside `recorded`, converted right away).
    """

    def __init__(self, dtype: mx.Dtype) -> None:
        """Convert to `dtype`."""
        self.dtype = dtype

    def a(self, a: np.ndarray) -> mx.array:
        """An array in the dtype, converted now (not through the cache)."""
        return mx.array(np.ascontiguousarray(a)).astype(self.dtype)

    def t(self, shape: tuple[int, ...], make: Callable[[], np.ndarray], dtype: mx.Dtype | None = None) -> mx.array:
        """The array of `shape` that `make()` computes from weight values (cached: `make` not run on a hit)."""
        dt = self.dtype if dtype is None else dtype

        def convert() -> mx.array:
            a = make()
            if tuple(a.shape) != shape:
                msg = f"igev mlx: built {a.shape}, declared {shape}"
                raise AssertionError(msg)
            return mx.array(np.ascontiguousarray(a, np.float32)).astype(dt)

        return mlx_port.tensor(SimpleNamespace(shape=shape), dt, convert)

    def pair(self, m: Any, norm: Any, perm: tuple[int, ...], axis: int = 0) -> tuple[mx.array, mx.array]:  # noqa: ANN401
        """A convolution's (folded) weight permuted by `perm` and its bias."""
        fold = functools.cache(lambda: _folded(m, norm, axis))
        shape = tuple(int(n) for n in m.weight.shape)
        return (
            self.t(tuple(shape[i] for i in perm), lambda: fold()[0].transpose(perm)),
            self.t((shape[axis],), lambda: fold()[1]),
        )

    def conv(self, m: Any, norm: Any = None) -> Conv:  # noqa: ANN401 - torch modules of the vendored net
        """`nn.Conv2d` / `nn.Conv3d` (and a batch norm after it)."""
        nd = m.weight.ndim
        w, b = self.pair(m, norm, (0, *range(2, nd), 1))
        return w, b, _ints(m.stride), _ints(m.padding)

    def deconv(self, m: Any, norm: Any = None) -> Conv:  # noqa: ANN401
        """`nn.ConvTranspose2d`: weight `[Cout, kh, kw, Cin]`."""
        w, b = self.pair(m, norm, (1, 2, 3, 0), axis=1)
        return w, b, int(m.stride[0]), int(m.padding[0])

    def basic(self, m: Any) -> tuple[Conv, bool, bool]:  # noqa: ANN401
        """`BasicConv` (batch norm folded) or `BasicConv_IN` (instance norm kept): (conv, instance norm, act)."""
        inorm = bool(getattr(m, "use_in", False))
        bn = m.bn if getattr(m, "use_bn", False) else None
        is3 = m.conv.weight.ndim == 5  # noqa: PLR2004
        if isinstance(m.conv, _transposed()):
            if is3:
                w, b = self.pair(m.conv, bn, (0, 1, 2, 3, 4), axis=1)
                return (w, b, int(m.conv.stride[0]), int(m.conv.padding[0])), inorm, bool(m.relu)
            return self.deconv(m.conv, bn), inorm, bool(m.relu)
        return self.conv(m.conv, bn), inorm, bool(m.relu)

    def conv2x(self, m: Any) -> tuple[Any, Any]:  # noqa: ANN401
        """`Conv2x` / `Conv2x_IN` (transposed, concatenated)."""
        return self.basic(m.conv1), self.basic(m.conv2)

    def att(self, m: Any) -> tuple[Any, Conv]:  # noqa: ANN401
        """`FeatureAtt`."""
        return self.basic(m.feat_att[0]), self.conv(m.feat_att[1])

    def res(self, m: Any) -> tuple[Conv, Conv, Conv | None]:  # noqa: ANN401
        """`ResidualBlock` (batch norms)."""
        down = None if m.downsample is None else self.conv(m.downsample[0], m.downsample[1])
        return self.conv(m.conv1, m.norm1), self.conv(m.conv2, m.norm2), down

    def dw(self, c: Any, n: Any = None) -> tuple[mx.array, mx.array, int]:  # noqa: ANN401
        """A depthwise convolution: `([kh, kw, C], bias, stride)`."""
        fold = functools.cache(lambda: _folded(c, n))
        ch, _, kh, kw = (int(v) for v in c.weight.shape)
        w = self.t((kh, kw, ch), lambda: fold()[0][:, 0].transpose(1, 2, 0))
        return w, self.t((ch,), lambda: fold()[1]), int(c.stride[0])

    def mobile(self, m: Any) -> tuple[Any, ...]:  # noqa: ANN401
        """A MobileNetV2 block: (expand or None, depthwise `([kh, kw, C], bias, stride)`, project, skip)."""
        if hasattr(m, "conv_pwl"):
            return self.conv(m.conv_pw, m.bn1), self.dw(m.conv_dw, m.bn2), self.conv(m.conv_pwl, m.bn3), m.has_skip
        return None, self.dw(m.conv_dw, m.bn1), self.conv(m.conv_pw, m.bn2), m.has_skip

    def feature(self, f: Any) -> dict[str, Any]:  # noqa: ANN401 - extractor.Feature
        """`Feature`: the MobileNetV2 stem and stages, the three up-sampling merges and `conv4`."""
        return {
            "stem": self.conv(f.conv_stem, f.bn1),
            "blocks": [[self.mobile(b) for stage in getattr(f, f"block{i}") for b in stage] for i in range(5)],
            "up": [self.conv2x(m) for m in (f.deconv32_16, f.deconv16_8, f.deconv8_4)],
            "conv4": self.basic(f.conv4),
        }

    def context(self, e: Any, convs: Any) -> dict[str, Any]:  # noqa: ANN401 - extractor.MultiBasicEncoder
        """`MultiBasicEncoder` (batch norms) and the `context_zqr_convs` after its `inp` heads."""
        return {
            "conv1": self.conv(e.conv1, e.norm1),
            "layers": [[self.res(b) for b in getattr(e, f"layer{i}")] for i in range(1, 6)],
            "out04": [(self.res(h[0]), self.conv(h[1])) for h in e.outputs04],
            "out08": [(self.res(h[0]), self.conv(h[1])) for h in e.outputs08],
            "out16": [self.conv(h) for h in e.outputs16],
            "zqr": [self.conv(c) for c in convs],
        }

    def hourglass(self, m: Any) -> dict[str, Any]:  # noqa: ANN401 - the families' `hourglass`
        """The three-level `hourglass`."""
        out: dict[str, Any] = {}
        for name, sub in m.named_children():
            if name.startswith("feature_att"):
                out[name] = self.att(sub)
            elif isinstance(sub, _sequential()):
                out[name] = [self.basic(c) for c in sub]
            else:
                out[name] = self.basic(sub)
        return out

    def gru(self, m: Any) -> tuple[Conv, Conv]:  # noqa: ANN401 - update.ConvGRU
        """`ConvGRU`: the z and r convolutions as one (`[z | r]` out), the q convolution."""
        o, i, kh, kw = (int(v) for v in m.convz.weight.shape)
        fz = functools.cache(lambda: _folded(m.convz, None))
        fr = functools.cache(lambda: _folded(m.convr, None))
        w = self.t((2 * o, kh, kw, i), lambda: np.concatenate([fz()[0], fr()[0]]).transpose(0, 2, 3, 1))
        b = self.t((2 * o,), lambda: np.concatenate([fz()[1], fr()[1]]))
        return (w, b, 1, int(m.convz.padding[0])), self.conv(m.convq)

    def gru_split(self, m: Any, hidden: int, inputs: int) -> tuple[tuple[Conv, Conv], Conv]:  # noqa: ANN401
        """A plain ConvGRU (`convz`, `convr`, `convq` with biases) whose input starts with a constant map.

        The input channels are `[h (hidden) | constant (inputs) | rest]`: returns the `gru` weights of `[h | rest]`
        (no bias) and one convolution of the constant to `[z | r | q]` with the biases, the `ctx` of `gru`.
        """
        folds = [functools.cache(lambda c=c: _folded(c, None)) for c in (m.convz, m.convr, m.convq)]
        o, i, kh, kw = (int(v) for v in m.convz.weight.shape)
        pad = int(m.convz.padding[0])
        rest = i - inputs

        def keep(w: np.ndarray) -> np.ndarray:
            return np.concatenate([w[:, :hidden], w[:, hidden + inputs :]], 1).transpose(0, 2, 3, 1)

        zero = mx.zeros((hidden,), self.dtype)
        wzr = self.t((2 * o, kh, kw, rest), lambda: np.concatenate([keep(folds[0]()[0]), keep(folds[1]()[0])]))
        zr = (wzr, mx.zeros((2 * hidden,), self.dtype), 1, pad)
        wq = self.t((o, kh, kw, rest), lambda: keep(folds[2]()[0]))
        ctx = self.t(
            (3 * o, kh, kw, inputs),
            lambda: np.concatenate([f()[0][:, hidden : hidden + inputs] for f in folds]).transpose(0, 2, 3, 1),
        )
        bias = self.t((3 * o,), lambda: np.concatenate([f()[1] for f in folds]))
        return (zr, (wq, zero, 1, pad)), (ctx, bias, 1, pad)


def _transposed() -> tuple[type, ...]:
    import torch  # noqa: PLC0415 - only to classify the vendored modules while copying

    return (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)


def _sequential() -> type:
    import torch  # noqa: PLC0415

    return torch.nn.Sequential


def arrays(*trees: object) -> list[mx.array]:
    """Every `mx.array` in nested lists / tuples / dicts (to evaluate the copied weights once)."""
    found: list[mx.array] = []

    def walk(v: object) -> None:
        if isinstance(v, mx.array):
            found.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(list(trees))
    return found


# layers ---------------------------------------------------------------------------------------------------------------


def conv(x: mx.array, c: Conv) -> mx.array:
    """A 2-D or 3-D convolution with its bias."""
    w, b, stride, padding = c
    if w.ndim == 5:  # noqa: PLR2004
        return mx.conv3d(x, w, stride=stride, padding=padding) + b
    return mx.conv2d(x, w, stride=stride, padding=padding) + b


def deconv(x: mx.array, c: Conv) -> mx.array:
    """A transposed 2-D convolution (`[Cout, kh, kw, Cin]`) or 3-D one (`[Cin, Cout, kd, kh, kw]`, stride 2)."""
    w, b, stride, padding = c
    if w.ndim == 5:  # noqa: PLR2004
        if not isinstance(stride, int) or not isinstance(padding, int):  # pragma: no cover - the nets' are uniform
            msg = f"igev mlx: a 3-D transposed convolution with per-axis stride {stride} / padding {padding}"
            raise TypeError(msg)
        k = w.shape[2] - 2 * padding - stride  # output padding of an exact stride-x up-sampling
        y = _conv_transpose3d_phases(x, w, (stride,) * 3, (padding,) * 3, (k,) * 3)
        if y is None:  # pragma: no cover - the pair is padded to 32s, every volume level halves exactly
            msg = f"igev mlx: the cost volume {tuple(x.shape[1:4])} does not double evenly"
            raise ValueError(msg)
        return y + b
    return mx.conv_transpose2d(x, w, stride=stride, padding=padding) + b


def basic(x: mx.array, m: tuple[Conv, bool, bool], *, transposed: bool = False) -> mx.array:
    """`BasicConv` / `BasicConv_IN`: convolution, instance norm if any, LeakyReLU if any."""
    c, norm, act = m
    y = deconv(x, c) if transposed else conv(x, c)
    if norm:
        y = inorm(y)
    return leaky(y) if act else y


def _conv2x(x: mx.array, rem: mx.array, m: tuple[Any, Any]) -> mx.array:
    """`Conv2x` / `Conv2x_IN` (transposed, concatenated): the pair is padded to 32s, so the sizes match."""
    return basic(mx.concatenate([basic(x, m[0], transposed=True), rem], axis=-1), m[1])


def _mobile(x: mx.array, blocks: list[tuple[Any, ...]]) -> mx.array:
    for expand, (w, b, stride), project, skip in blocks:
        y = x if expand is None else _relu6(conv(x, expand))
        y = _relu6(depthwise(y, w, b, (stride, stride), (1, 1), (1, 1)))
        y = conv(y, project)
        x = x + y if skip else y
    return x


def feature(x: mx.array, f: dict[str, Any]) -> list[mx.array]:
    """`Feature`: `[x4 (48), x8 (64), x16 (192), x32 (160)]` of NHWC images."""
    x = _relu6(conv(x, f["stem"]))
    x2 = _mobile(x, f["blocks"][0])
    x4 = _mobile(x2, f["blocks"][1])
    x8 = _mobile(x4, f["blocks"][2])
    x16 = _mobile(x8, f["blocks"][3])
    x32 = _mobile(x16, f["blocks"][4])
    x16 = _conv2x(x32, x16, f["up"][0])
    x8 = _conv2x(x16, x8, f["up"][1])
    x4 = basic(_conv2x(x8, x4, f["up"][2]), f["conv4"])
    return [x4, x8, x16, x32]


def _res(x: mx.array, m: tuple[Conv, Conv, Conv | None]) -> mx.array:
    y = relu(conv(relu(conv(x, m[0])), m[1]))
    return relu((x if m[2] is None else conv(x, m[2])) + y)


def context(x: mx.array, e: dict[str, Any], levels: int = 3) -> list[tuple[mx.array, list[mx.array]]]:
    """`MultiBasicEncoder` + `context_zqr_convs`: per level `(tanh(net), [cz | cr, cq] of relu(inp))`, 1/4 first."""
    x = relu(conv(x, e["conv1"]))
    for layer in e["layers"][:3]:
        for b in layer:
            x = _res(x, b)
    outs = [[conv(_res(x, r), c) for r, c in e["out04"]]]
    if levels > 1:
        for b in e["layers"][3]:
            x = _res(x, b)
        outs.append([conv(_res(x, r), c) for r, c in e["out08"]])
    if levels > 2:  # noqa: PLR2004
        for b in e["layers"][4]:
            x = _res(x, b)
        outs.append([conv(x, c) for c in e["out16"]])
    if not e["zqr"]:  # no context convolutions: (tanh(net), [relu(inp)])
        return [(mx.tanh(net), [relu(inp)]) for net, inp in outs]
    result = []
    for (net, inp), zqr in zip(outs, e["zqr"], strict=False):
        c = conv(relu(inp), zqr)
        k = c.shape[-1] // 3
        result.append((mx.tanh(net), [c[..., : 2 * k], c[..., 2 * k :]]))
    return result


def gate(cost: mx.array, im: mx.array, att: tuple[Any, Conv]) -> mx.array:
    """`FeatureAtt`: `cost` `[B, D, H, W, C]` gated by image features `[B, H, W, K]`."""
    g = mx.sigmoid(conv(basic(im, att[0]), att[1])).astype(cost.dtype)
    return cost * g[:, None]


def hourglass(x: mx.array, feats: list[mx.array], m: dict[str, Any]) -> mx.array:
    """The families' `hourglass`: `x` `[B, D, H, W, C]` at 1/4, gated by `feats[1..3]` (1/8 .. 1/32)."""

    def seq(x: mx.array, name: str) -> mx.array:
        for c in m[name]:
            x = basic(x, c)
        return x

    conv1 = gate(seq(x, "conv1"), feats[1], m["feature_att_8"])
    conv2 = gate(seq(conv1, "conv2"), feats[2], m["feature_att_16"])
    conv3 = gate(seq(conv2, "conv3"), feats[3], m["feature_att_32"])
    conv2 = mx.concatenate([basic(conv3, m["conv3_up"], transposed=True), conv2], axis=-1)
    conv2 = gate(seq(conv2, "agg_0"), feats[2], m["feature_att_up_16"])
    conv1 = mx.concatenate([basic(conv2, m["conv2_up"], transposed=True), conv1], axis=-1)
    conv1 = gate(seq(conv1, "agg_1"), feats[1], m["feature_att_up_8"])
    return basic(conv1, m["conv1_up"], transposed=True)


def pool2x(x: mx.array) -> mx.array:
    """`F.avg_pool2d(x, 3, stride=2, padding=1)` (zeros counted) of an NHWC map, as the shared depthwise kernel."""
    c = x.shape[-1]
    w = mx.full((3, 3, c), 1.0 / 9.0, dtype=x.dtype)
    return depthwise(x, w, mx.zeros((c,), x.dtype), (2, 2), (1, 1), (1, 1))


def interp(x: mx.array, size: tuple[int, int]) -> mx.array:
    """`interp`: bilinear, align_corners=True, to `size` (float32 weights, one rounding)."""
    h, w = x.shape[1:3]
    ratios = ((h - 1) / max(size[0] - 1, 1), (w - 1) / max(size[1] - 1, 1))
    return bilinear_nhwc(x, size, ratios, align=True)


def gru(h: mx.array, ctx: list[mx.array], x: mx.array, m: tuple[Conv, Conv]) -> mx.array:
    """`ConvGRU(h, cz, cr, cq, *x)` with `x` already concatenated; `ctx[0]` is `[cz | cr]` (one array)."""
    zr = mx.sigmoid(conv(mx.concatenate([h, x], axis=-1), m[0]) + ctx[0])
    k = h.shape[-1]
    z, r = zr[..., :k], zr[..., k:]
    q = mx.tanh(conv(mx.concatenate([r * h, x], axis=-1), m[1]) + ctx[1])
    return h + z * (q - h)


def context_upsample(disp: mx.array, prob: mx.array) -> mx.array:
    """`context_upsample`: every pixel of `prob` `[1, 4H, 4W, 9]` mixes the 3 x 3 neighbourhood of its coarse pixel."""
    h, w = disp.shape
    dp = mx.pad(disp, [(1, 1), (1, 1)])
    near = mx.stack([dp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=-1)  # [H, W, 9]
    return (prob.reshape(h, 4, w, 4, 9) * near[:, None, :, None]).sum(axis=-1).reshape(1, 4 * h, 4 * w)


def _pool2_last(x: mx.array) -> mx.array:
    n = x.shape[-1] // 2
    return 0.5 * (x[..., 0 : 2 * n : 2] + x[..., 1 : 2 * n : 2])


class GeoLookup:
    """`FastGeoEncodingVolume`: geometry volume `[P, C, D]` float32 and all-pairs row correlation `[P, W]` pyramids."""

    def __init__(  # noqa: PLR0913, PLR0917
        self, f1: mx.array, f2: mx.array, geo: mx.array, levels: int, radius: int, corr_dtype: mx.Dtype
    ) -> None:
        """`f1`, `f2` `[1, H, W, C]` matching features, `geo` `[1, D, H, W, C]` (NDHWC)."""
        _, d, h, w, c = geo.shape
        g = geo[0].transpose(1, 2, 3, 0).reshape(h * w, c, d).astype(mx.float32)
        a, b = f1[0].astype(corr_dtype), f2[0].astype(corr_dtype)
        corr = (a @ b.transpose(0, 2, 1)).reshape(h * w, w)  # [H, W, C] @ [H, C, W2]
        self.geo, self.corr = [g], [corr]
        for _ in range(levels - 1):
            g = _pool2_last(g)
            corr = _pool2_last(corr)
            self.geo.append(g)
            self.corr.append(corr)
        self.dx = mx.arange(-radius, radius + 1, dtype=mx.float32)[None]
        self.coords = mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None, :], (h, w)).reshape(-1, 1)

    def state(self) -> list[mx.array]:
        """The pyramids and the lookup offsets: the inputs `lookup` takes (a compiled step's, not its constants)."""
        return [*self.geo, *self.corr, self.coords, self.dx]

    @staticmethod
    def lookup(disp: mx.array, state: list[mx.array]) -> mx.array:
        """`[1, H, W, levels * (C + 1) * K]` float32 lookups around `disp` `[1, H, W, 1]` (float32) in `state`."""
        n = (len(state) - 2) // 2
        geo, corr, coords, dx = state[:n], state[n : 2 * n], state[-2], state[-1]
        h, w = disp.shape[1:3]
        dsp = disp.reshape(-1, 1)
        out = []
        for i in range(n):
            sc = 1.0 / 2**i
            out.append(linear_sample(geo[i], dsp * sc + dx))
            out.append(linear_sample(corr[i], coords * sc - dsp * sc + dx))
        return mx.concatenate(out, axis=1).reshape(1, h, w, -1)
