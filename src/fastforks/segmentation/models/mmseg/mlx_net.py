"""The mmseg CNN classics on Apple MLX, written in `mlx.core` operations (`MlxMMSeg(net, dtype)`): image -> logits.

`EncoderDecoder.forward` of the BatchNorm-folded network (`model.fold_batchnorms`), channels last. Written out
instead of replayed from an exported graph: the export of ResNet-101 (HRNet-W48: ~300 convs) was 1.1-3.3 s of a first
call whose network runs in 0.2-0.7 s.

- convolutions dispatch as the graph translator's (`shared.mlx_graph`): depthwise ones on the shared depthwise kernel,
  1x1 ones as GEMMs (`conv1x1_ok`), the rest through `mlx_graph._conv2d` (its channel padding onto MLX's fast tiles);
  kernels mode adds the `conv_mlx` Winograd kernel for stride-1 3x3 convs and its direct kernel for the RGB stems;
- bias, residual and ReLU after an `mx.conv2d` or a GEMM are one compiled elementwise pass;
- the pooled branches (PSP bins, image pooling) average in float32, as the translator's adaptive pooling;
- bilinear resizes are `shared.mlx_seg.resize` (torch's half-pixel values).

The network is evaluated block by block (`_step`), one command buffer in flight: MLX returns the buffers of a command
buffer when it completes, so one lazy graph of the whole network (or of a head with several MLX Winograd convs) holds
every transient at once, and unbounded `mx.async_eval` lets Python encode blocks ahead of the GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared.mlx_seg import resize

from . import net as mm

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MlxMMSeg"]


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


@mx.compile
def _bias_relu(y: mx.array, b: mx.array) -> mx.array:
    return mx.maximum(y + b, 0)


@mx.compile
def _bias_res_relu(y: mx.array, b: mx.array, r: mx.array) -> mx.array:
    return mx.maximum(y + b + r, 0)


@mx.compile
def _res_relu(y: mx.array, r: mx.array) -> mx.array:
    return mx.maximum(y + r, 0)


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return np.ascontiguousarray(t.detach().float().numpy())


def _pair(v: Any) -> tuple[int, int]:  # noqa: ANN401 - an int or a pair
    return (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))


class _Conv:
    """A (folded) `Conv2d` on NHWC maps: `conv(x, relu=, r=)` is `relu?(conv(x) + bias + r)`."""

    def __init__(self, conv: Any, dtype: mx.Dtype, *, kernels: bool) -> None:  # noqa: ANN401 - a torch `Conv2d`
        w = _np(conv.weight)  # [O, I/g, kh, kw]
        self.cout, cin_g, kh, kw = w.shape
        self.k = (kh, kw)
        self.stride, self.padding, self.dilation = _pair(conv.stride), _pair(conv.padding), _pair(conv.dilation)
        self.groups = conv.groups
        self.cin = cin_g * self.groups
        b = _np(conv.bias) if conv.bias is not None else np.zeros(self.cout, np.float32)
        self.b = mx.array(b).astype(dtype)
        self.kernels = kernels
        self.dw = self.wm = self.wl = None
        self.u: dict[int, mx.array] = {}  # Winograd-domain weights per tile (kernels mode)
        if self.groups > 1:
            if not (self.groups == self.cin == self.cout and self.cout % 4 == 0 and kh * kw > 1):
                msg = f"grouped conv {self.cin} -> {self.cout} (groups {self.groups})"
                raise NotImplementedError(msg)
            self.dw = mx.array(np.ascontiguousarray(w[:, 0].transpose(1, 2, 0))).astype(dtype)  # [kh, kw, C]
            return
        if mg.conv1x1_ok(self.cin, self.cout, 0, self.k, self.stride, self.padding, 1):
            self.wm = mx.array(np.ascontiguousarray(w[:, :, 0, 0].T)).astype(dtype)  # [in, out]
            return
        self.wl = mx.array(np.ascontiguousarray(w.transpose(0, 2, 3, 1))).astype(dtype)  # OHWI
        self.wino = (
            kernels
            and self.k == (3, 3)
            and self.stride == (1, 1)
            and self.padding == (1, 1)
            and self.dilation == (1, 1)
            and self.cin <= 1024  # noqa: PLR2004 - as `mlx_graph._wino_ok`
            and self.cout <= 1024  # noqa: PLR2004
        )
        self.direct = kernels and not mg.DIRECT_OFF and _direct_supported(self.cin, self.cout, self.k, self.stride)

    def _winograd(self, x: mx.array) -> mx.array:
        """`mlx_graph._wino_weights`: small layers keep their transformed weights, big ones make them per call."""
        from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - kernels mode only

        assert self.wl is not None  # noqa: S101 - a dense conv
        m = mg._wino_tile(x)  # noqa: SLF001 - the translator's tile choice
        if (m + 2) ** 2 * self.cin * self.cout * x.dtype.size > mg.WINO_LAZY_MB * (1 << 20):
            return conv_ops.weights_graph(self.wl, m, x.dtype)
        if m not in self.u:
            self.u[m] = conv_ops.weights(self.wl, m, x.dtype)
        return self.u[m]

    def __call__(self, x: mx.array, *, relu: bool = False, r: mx.array | None = None) -> mx.array:
        """NHWC `x` -> NHWC `relu?(conv(x) + bias [+ r])`."""
        if self.dw is not None:
            y = ops.depthwise(x, self.dw, self.b, self.stride, self.padding, self.dilation)
            return _epilogue(y, None, relu=relu, r=r)
        if self.wm is not None:
            if not relu and r is None:
                return mg.conv1x1(x, self.wm, self.b)
            n, h, w, _ = x.shape
            y = (x.reshape(-1, self.cin) @ self.wm).reshape(n, h, w, self.cout)
            return _epilogue(y, self.b, relu=relu, r=r)
        assert self.wl is not None  # noqa: S101 - a dense conv
        if self.kernels:
            from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - kernels mode only

            if self.direct and x.shape[1] * x.shape[2] >= mg.DIRECT_MIN_HW:
                y = conv_ops.direct(
                    x, self.wl, self.b, self.stride, self.padding, self.dilation, act=int(relu and r is None)
                )
                return y if r is None else _epilogue(y, None, relu=relu, r=r)
            if self.wino and conv_ops.supported(tuple(x.shape), self.cout):
                m = mg._wino_tile(x)  # noqa: SLF001
                y = conv_ops.conv3x3(x, self._winograd(x), self.b, m=m, act=int(relu and r is None))
                return y if r is None else _epilogue(y, None, relu=relu, r=r)
        y = mg._conv2d(x, self.wl, self.stride, self.padding, self.dilation, 1)  # noqa: SLF001 - the channel padding
        return _epilogue(y, self.b, relu=relu, r=r)


def _direct_supported(cin: int, cout: int, k: tuple[int, int], stride: tuple[int, int]) -> bool:
    from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - kernels mode only

    return conv_ops.direct_supported(cin, cout, k, stride)


def _epilogue(y: mx.array, b: mx.array | None, *, relu: bool, r: mx.array | None) -> mx.array:
    if b is not None:
        if relu:
            return _bias_relu(y, b) if r is None else _bias_res_relu(y, b, r)
        y = y + b
    if r is not None:
        return _res_relu(y, r) if relu else y + r
    return _relu(y) if relu else y


def _chain(seq: Any, dtype: mx.Dtype, *, kernels: bool) -> list[tuple[_Conv, bool]]:  # noqa: ANN401 - a Sequential
    """The convs of a folded `Sequential` (nested ones flattened), each with whether a ReLU follows it."""
    out: list[tuple[_Conv, bool]] = []
    for m in seq.children():
        name = type(m).__name__
        if name == "Conv2d":
            out.append((_Conv(m, dtype, kernels=kernels), False))
        elif name == "ReLU":
            out[-1] = (out[-1][0], True)
        elif name == "Sequential":
            out += _chain(m, dtype, kernels=kernels)
        elif name != "Identity":  # a folded BatchNorm
            msg = f"unexpected module {name} in a conv chain"
            raise NotImplementedError(msg)
    return out


def _run_chain(x: mx.array, chain: list[tuple[_Conv, bool]]) -> mx.array:
    for conv, relu in chain:
        x = conv(x, relu=relu)
    return x


def _bins(n: int, s: int) -> mx.array:
    """`[s, n]` float32 averaging matrix of adaptive pooling (`net._bins`)."""
    m = np.zeros((s, n), np.float32)
    for i in range(s):
        a, b = i * n // s, -(-(i + 1) * n // s)
        m[i, a:b] = 1 / (b - a)
    return mx.array(m)


def _adaptive_avg(x: mx.array, s: int) -> mx.array:
    """`adaptive_avg_pool2d(x, s)` of NHWC `x`, in float32, back to its dtype."""
    n, h, w, c = x.shape
    x32 = x.astype(mx.float32)
    if s == 1:
        return mx.mean(x32, axis=(1, 2), keepdims=True).astype(x.dtype)
    t = (_bins(h, s) @ x32.reshape(n, h, w * c)).reshape(n, s, w, c)  # [n, s, w, c]
    return (_bins(w, s) @ t).astype(x.dtype)  # [s, w] @ [n, s, w, c] -> [n, s, s, c]


def _max_pool(x: mx.array) -> mx.array:
    """`max_pool2d(x, 3, 2, 1)` of NHWC `x` after a ReLU (zero padding is then exact): nine strided slices."""
    h, w = x.shape[1:3]
    oh, ow = (h - 1) // 2 + 1, (w - 1) // 2 + 1
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    y = None
    for i in range(3):
        for j in range(3):
            v = xp[:, i : i + 2 * oh - 1 : 2, j : j + 2 * ow - 1 : 2]
            y = v if y is None else mx.maximum(y, v)
    assert y is not None  # noqa: S101
    return y


def _resize(x: mx.array, size: Sequence[int]) -> mx.array:
    """Bilinear (half-pixel) NHWC resize in `x`'s dtype."""
    if tuple(x.shape[1:3]) == tuple(size):
        return x
    return resize(x, size).astype(x.dtype)


class _Block:
    """A ResNet `Bottleneck` / `BasicBlock` (folded): the convs, the shortcut, the residual ReLU."""

    def __init__(self, blk: Any, dtype: mx.Dtype, *, kernels: bool) -> None:  # noqa: ANN401 - a torch block
        names = ("conv1", "conv2", "conv3") if hasattr(blk, "conv3") else ("conv1", "conv2")
        self.convs = [_Conv(getattr(blk, n), dtype, kernels=kernels) for n in names]
        self.down = None if blk.downsample is None else _chain(blk.downsample, dtype, kernels=kernels)

    def __call__(self, x: mx.array) -> mx.array:
        idt = x if self.down is None else _run_chain(x, self.down)
        for conv in self.convs[:-1]:
            x = conv(x, relu=True)
        return self.convs[-1](x, relu=True, r=idt)


def _blocks(seq: Any, dtype: mx.Dtype, *, kernels: bool) -> list[_Block]:  # noqa: ANN401 - a Sequential of blocks
    return [_Block(b, dtype, kernels=kernels) for b in seq]


_PENDING: list[Any] = []  # the step in flight (`_step`)


def _step(x: Any) -> Any:  # noqa: ANN401 - an array or a list of them
    """Schedule `x` and wait for the step before it (pipeline depth 1: the GPU stays busy, the transients bounded).

    fcn-r101 float32, 512x1024: unbounded `async_eval` 416 ms / 1848 MB MLX peak, depth 1 417 ms / 1756 MB, fully
    synchronous 440 ms / 1660 MB.
    """
    mx.async_eval(x)
    if _PENDING:
        mx.eval(_PENDING.pop())
    _PENDING.append(x)
    return x


def _run_blocks(x: mx.array, blocks: list[_Block]) -> mx.array:
    for blk in blocks:
        x = _step(blk(x))
    return x


class _HRModule:
    """`HRModule.forward`: per-branch blocks, then each output sums all branches (resampled), ReLU."""

    def __init__(self, m: Any, dtype: mx.Dtype, *, kernels: bool) -> None:  # noqa: ANN401 - an `HRModule`
        self.branches = [_blocks(br, dtype, kernels=kernels) for br in m.branches]
        self.fuse: list[list[Any]] | None = None
        if m.fuse_layers is None:
            return
        self.fuse = []
        for i, row in enumerate(m.fuse_layers):
            terms: list[Any] = []
            for j, f in enumerate(row):
                if j == i:
                    terms.append(None)
                elif j > i:
                    terms.append((_Conv(f[0], dtype, kernels=kernels), float(f[2].scale)))
                else:
                    terms.append(_chain(f, dtype, kernels=kernels))
            self.fuse.append(terms)

    def __call__(self, xs: list[mx.array]) -> list[mx.array]:
        xs = [_run_blocks(x, br) for x, br in zip(xs, self.branches, strict=True)]
        if self.fuse is None:
            return xs
        out = []
        for i, terms in enumerate(self.fuse):
            y = None
            for j, f in enumerate(terms):
                if j == i:
                    t = xs[j]
                elif j > i:
                    conv, scale = f
                    z = conv(xs[j])
                    z = _resize(z, [int(s * scale) for s in z.shape[1:3]])  # mmseg `Upsample`: int(size * scale)
                    t = _resize(z, xs[i].shape[1:3])
                else:
                    t = _run_chain(xs[j], f)
                y = t if y is None else y + t
            assert y is not None  # noqa: S101
            out.append(_relu(y))
        return _step(out)


class _ResNet:
    """`ResNetV1c`: the deep stem, max pooling, four stages; the four stage outputs."""

    def __init__(self, b: mm.ResNetV1c, dtype: mx.Dtype, *, kernels: bool) -> None:
        self.stem = _chain(b.stem, dtype, kernels=kernels)
        self.layers = [_blocks(getattr(b, n), dtype, kernels=kernels) for n in b.layers]

    def __call__(self, x: mx.array) -> list[mx.array]:
        x = _max_pool(_run_chain(x, self.stem))
        outs = []
        for layer in self.layers:
            x = _run_blocks(x, layer)
            outs.append(x)
        return outs


class _HRNet:
    """`HRNet`: the stem, the bottleneck stage, then stages of 2 / 3 / 4 branches with their transitions."""

    def __init__(self, b: mm.HRNet, dtype: mx.Dtype, *, kernels: bool) -> None:
        self.stem = [_Conv(b.conv1, dtype, kernels=kernels), _Conv(b.conv2, dtype, kernels=kernels)]
        self.layer1 = _blocks(b.layer1, dtype, kernels=kernels)
        self.stages = []
        for s in (2, 3, 4):
            trans = [None if t is None else _chain(t, dtype, kernels=kernels) for t in getattr(b, f"transition{s - 1}")]
            mods = [_HRModule(m, dtype, kernels=kernels) for m in getattr(b, f"stage{s}")]
            self.stages.append((trans, mods))

    def __call__(self, x: mx.array) -> list[mx.array]:
        for conv in self.stem:
            x = conv(x, relu=True)
        ys = [_run_blocks(x, self.layer1)]
        for trans, mods in self.stages:
            ys = [
                ys[i] if t is None else _run_chain(ys[-1] if i >= len(ys) - 1 else ys[i], t)
                for i, t in enumerate(trans)
            ]
            for m in mods:
                ys = m(ys)
        return ys


class MlxMMSeg:
    """The folded network `net` (eval, float32 weights) in `dtype` (the name); `kernels`: our Metal conv kernels."""

    def __init__(self, net: mm.EncoderDecoder, dtype: str, *, kernels: bool) -> None:
        """Convert the weights of `net`."""
        self.dtype = getattr(mx, dtype)
        self.kernels = kernels
        dt, kw = self.dtype, {"kernels": kernels}
        b: Any = net.backbone  # the vendored modules type their submodules as `Module`
        head: Any = net.decode_head

        def cm(m: Any) -> _Conv:  # noqa: ANN401 - a `ConvModule` (its BN folded, its ReLU `m.act`)
            return _Conv(m.conv, dt, **kw)

        def ds(m: Any) -> tuple[_Conv, _Conv]:  # noqa: ANN401 - a `DepthwiseSeparableConvModule`
            return cm(m.depthwise_conv), cm(m.pointwise_conv)

        self.backbone: Any = _HRNet(b, dt, **kw) if isinstance(b, mm.HRNet) else _ResNet(b, dt, **kw)
        self.head = type(head).__name__
        self.conv_seg = _Conv(head.conv_seg, dt, **kw)
        if isinstance(head, mm.FCNHead):
            self.in_index = head.in_index
            self.convs = [cm(m) for m in head.convs]
            self.conv_cat = None if head.conv_cat is None else cm(head.conv_cat)
        elif isinstance(head, mm.PSPHead):
            self.psp = [(int(m[0].s), cm(m[1])) for m in head.psp_modules]
            self.bottleneck = cm(head.bottleneck)
        elif isinstance(head, mm.ASPPHead):  # DeepLabV3 and DeepLabV3+
            self.image_pool = cm(head.image_pool[1])
            self.aspp = [ds(m) if isinstance(m, mm.DepthwiseSeparableConvModule) else cm(m) for m in head.aspp_modules]
            self.bottleneck = cm(head.bottleneck)
            if isinstance(head, mm.DepthwiseSeparableASPPHead):
                self.c1 = cm(head.c1_bottleneck)
                self.sep = [ds(m) for m in head.sep_bottleneck]
        else:
            msg = f"mmseg head {self.head}"
            raise NotImplementedError(msg)

    @staticmethod
    def _cm(x: mx.array, m: _Conv | tuple[_Conv, _Conv]) -> mx.array:
        """A `ConvModule` (conv + ReLU) or a `DepthwiseSeparableConvModule` (two of them)."""
        if isinstance(m, tuple):
            return m[1](m[0](x, relu=True), relu=True)
        return m(x, relu=True)

    def _aspp(self, x: mx.array) -> mx.array:
        pooled = self.image_pool(_adaptive_avg(x, 1), relu=True)
        outs = [mx.broadcast_to(pooled, (*x.shape[:3], pooled.shape[-1]))]
        outs += [_step(self._cm(x, m)) for m in self.aspp]
        return _step(self.bottleneck(mx.concatenate(outs, axis=-1), relu=True))

    def _head(self, feats: list[mx.array]) -> mx.array:
        if self.head == "FCNHead":
            if isinstance(self.in_index, list):
                xs = [feats[i] for i in self.in_index]
                x = mx.concatenate([xs[0], *[_resize(t, xs[0].shape[1:3]) for t in xs[1:]]], axis=-1)
            else:
                x = feats[self.in_index]
            y = x
            for conv in self.convs:
                y = _step(conv(y, relu=True))
            if self.conv_cat is not None:
                y = _step(self.conv_cat(mx.concatenate([x, y], axis=-1), relu=True))
            return self.conv_seg(y)
        x = feats[3]
        if self.head == "PSPHead":
            size = x.shape[1:3]
            outs = [x, *[_resize(conv(_adaptive_avg(x, s), relu=True), size) for s, conv in self.psp]]
            return self.conv_seg(_step(self.bottleneck(mx.concatenate(_step(outs), axis=-1), relu=True)))
        y = self._aspp(x)
        if self.head == "ASPPHead":
            return self.conv_seg(y)
        c1 = self.c1(feats[0], relu=True)
        y = mx.concatenate([_resize(y, c1.shape[1:3]), c1], axis=-1)
        for m in self.sep:
            y = _step(self._cm(y, m))
        return self.conv_seg(y)

    def __call__(self, image: mx.array) -> mx.array:
        """Normalised image `[B, 3, H, W]` -> float32 logits `[B, classes, h, w]` at the head's resolution."""
        # the translator's conv options as its `GraphNet` sets them per call (module state `_conv2d` reads)
        mg._KERNELS, mg._GEMM1X1 = self.kernels, True  # noqa: SLF001
        mg._HALF_WINO_F32 = mg._HALF_WINO_PAD = False  # noqa: SLF001
        x = image.transpose(0, 2, 3, 1).astype(self.dtype)
        try:
            y = self._head(self.backbone(x))
        finally:
            _PENDING.clear()
        return y.astype(mx.float32).transpose(0, 3, 1, 2)
