"""CSPNeXt and its PAFPN neck on Apple MLX, written in `mlx.core` operations (RTMPose / RTMW, RTMDet).

The maths of the torch modules of this package, channels last, on a network with its batch norms folded:

- convs in the run dtype: 1 x 1 convs are matrix products, the 5 x 5 depthwise convs the shared depthwise kernel
  (`graph_mlx.ops.depthwise`; `mx.conv2d(groups=C)` falls off its fast path), the others `mx.conv2d`; the channel
  attention's spatial mean float32; the SPP's 9 x 9 and 13 x 13 max pools are the 5 x 5 one applied two and three
  times (same result: a max of maxes over the same window, the padding -inf);
- SiLU computed in float32: rtmw-l-384 float16 keypoints p99 1.30 px (tol 1.2) with a float16 SiLU, 0.79 px with it
  (YOLOX and RTMO build theirs in the run dtype, `run_dtype_silu`).

`graph="portable"` / `"kernels"` (the detectors) dispatches the dense convs as the graph translator does
(`shared.mlx_graph`): 1 x 1 ones as its GEMM, the others through `_conv2d` (zero channels onto MLX's fast GEMM tiles:
RTMDet-tiny's 48 / 96 widths), kernels mode first the `conv_mlx` Winograd 3 x 3 and direct few-channel convs. The
caller sets the translator's module options for the call (`options`). `graph=None` (RTMPose) keeps `mx.conv2d`.
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks._core.patch import base_type
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared import mlx_port

from . import SPP_POOLS

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    Layer = Callable[[mx.array], mx.array]

__all__ = [
    "arr",
    "backbone",
    "chain",
    "conv",
    "conv_act",
    "max_pool",
    "module",
    "neck",
    "options",
    "run_dtype_silu",
    "silu",
    "spp",
    "stepped",
    "steps",
    "up2",
]

POOL = SPP_POOLS[0]  # the SPP's smallest max pool, which the larger ones repeat (5, 9, 13)


def arr(t: Any, dtype: mx.Dtype) -> mx.array:
    """A torch tensor as an evaluated MLX array in `dtype` (via float64: no double rounding to float16).

    Read through `mlx_port.tensor`: a network built by `mlx_port.recorded` takes it from the conversion cache.
    """

    def convert() -> mx.array:
        a = mx.array(np.ascontiguousarray(t.detach().double().numpy(), dtype=np.float32)).astype(dtype)
        mx.eval(a)
        return a

    return mlx_port.tensor(t, dtype, convert)


def silu(x: mx.array) -> mx.array:
    """SiLU computed in float32 (fused into the conv's epilogue by `mx.compile`)."""
    y = x.astype(mx.float32)
    return (y * mx.sigmoid(y)).astype(x.dtype)


def _silu_run_dtype(x: mx.array) -> mx.array:
    """SiLU in the run dtype (`run_dtype_silu`)."""
    return x * mx.sigmoid(x)


_ACT = [silu]  # the SiLU `conv_act` builds with (`run_dtype_silu`)


@contextmanager
def run_dtype_silu() -> Generator[None]:
    """Layers built inside compute SiLU in the run dtype, as the exported networks did.

    YOLOX and RTMO: the float32 SiLU moved yolox-m / yolox-s / rtmo-t float16 just past their golden limits (boxes
    p99 0.511 of 0.51, scores p99 0.0049); the run-dtype one keeps all of them inside.
    """
    _ACT.append(_silu_run_dtype)
    try:
        yield
    finally:
        _ACT.pop()


def conv(layer: Any, dtype: mx.Dtype, graph: str | None = None) -> Layer:
    """A `Conv2d` (bias folded in) on NHWC maps: 1 x 1 as a matrix product, depthwise as the shared kernel."""
    stride, padding = (int(layer.stride[0]), int(layer.stride[1])), (int(layer.padding[0]), int(layer.padding[1]))
    w, b = layer.weight, arr(layer.bias, dtype)
    if layer.kernel_size == (1, 1) and stride == (1, 1) and layer.groups == 1:
        if graph is None:
            m = arr(w[:, :, 0, 0].t(), dtype)
            return lambda x: x @ m + b
        w1 = arr(w.permute(0, 2, 3, 1), dtype)  # one copy: the GEMM takes its transposed `[in, out]` view
        # the translator's option, read at trace time (`options`): a GEMM, or `mx.conv2d` (YOLOX, see its runners)
        return lambda x: (
            mg.conv1x1(x, w1.reshape(w1.shape[0], -1).T, b)
            if mg._GEMM1X1  # noqa: SLF001
            else mg._conv2d(x, w1, (1, 1), (0, 0), (1, 1), 1) + b  # noqa: SLF001
        )
    if layer.groups > 1 and layer.groups == layer.in_channels == layer.out_channels and layer.in_channels % 4 == 0:
        k = arr(w[:, 0].permute(1, 2, 0), dtype)  # [kh, kw, C]
        return lambda x: ops.depthwise(x, k, b, stride, padding, (1, 1))
    g = int(layer.groups)
    ohwi = arr(w.permute(0, 2, 3, 1), dtype)
    if graph is None:
        return lambda x: mx.conv2d(x, ohwi, stride=stride, padding=padding, groups=g) + b
    # the OIHW shape the translator's checks read, and the Winograd weight cache's key (`mlx_graph._wino_weights`):
    # a stand-in, not a second copy of the weights
    oihw: Any = SimpleNamespace(shape=tuple(w.shape))
    kernels = graph == "kernels"

    def apply(x: mx.array) -> mx.array:
        if kernels:
            from fastforks.kernels.conv_mlx import ops as conv_ops  # noqa: PLC0415 - kernels mode only

            if g == 1 and mg._direct_ok(x.shape[3], x.shape[1] * x.shape[2], oihw.shape, stride):  # noqa: SLF001
                return conv_ops.direct(x, ohwi, b, stride, padding, (1, 1))
            if mg._wino_ok(x, oihw, stride, padding, (1, 1), g):  # noqa: SLF001
                t = mg._wino_tile(x)  # noqa: SLF001
                return conv_ops.conv3x3(x, mg._wino_weights(oihw, ohwi, x.dtype, t), b, m=t)  # noqa: SLF001
        return mg._conv2d(x, ohwi, stride, padding, (1, 1), g) + b  # noqa: SLF001 - the translator's channel padding

    return apply


def options(graph: str | None, *, gemm1x1: bool = True, half_wino_f32: bool = False) -> None:
    """Set the translator's module options (`conv`, `mlx_graph._conv2d` read them) for a call of a `graph` network.

    The `mlx_wrap.port` options of the same names: 1 x 1 convs as GEMMs, float16 Winograd convs run in float32.
    """
    if graph is not None:
        mg._KERNELS, mg._GEMM1X1 = graph == "kernels", gemm1x1  # noqa: SLF001
        mg._HALF_WINO_F32, mg._HALF_WINO_PAD = half_wino_f32, False  # noqa: SLF001


_PENDING: list[Any] = []  # the step in flight (`stepped`)


def stepped(f: Callable[..., Any]) -> Callable[..., Any]:
    """`f` compiled, each result scheduled as a step, the step before it awaited (pipeline depth 1).

    MLX frees a command buffer's transients only when it completes: one compiled graph of a whole detector holds them
    all (rtmdet-x float32 at 800 x 1344: a call's footprint 1860 MB, the exported port's 1353-1492 MB). Run the steps
    inside `steps`.
    """
    g = mx.compile(f)

    def run(*args: Any) -> Any:
        y = g(*args)
        mx.async_eval(y)
        if _PENDING:
            mx.eval(_PENDING.pop())
        _PENDING.append(y)
        return y

    return run


def steps(f: Callable[..., Any], *args: Any) -> Any:
    """`f(*args)` with its `stepped` pieces, the pipeline emptied afterwards (also on an error)."""
    try:
        return f(*args)
    finally:
        _PENDING.clear()


def conv_act(layer: Any, dtype: mx.Dtype, graph: str | None = None) -> Layer:
    """`ConvAct`: SiLU(conv(x))."""
    c, act = conv(layer.conv, dtype, graph), _ACT[-1]
    return lambda x: act(c(x))


def chain(*layers: Layer) -> Layer:
    """The layers one after the other."""

    def apply(x: mx.array) -> mx.array:
        for op in layers:
            x = op(x)
        return x

    return apply


def module(layer: Any, dtype: mx.Dtype, graph: str | None = None) -> Layer:
    """`ConvAct` or `Separable` (depthwise + SiLU, pointwise + SiLU)."""
    if hasattr(layer, "depthwise_conv"):
        return chain(conv_act(layer.depthwise_conv, dtype, graph), conv_act(layer.pointwise_conv, dtype, graph))
    return conv_act(layer, dtype, graph)


def _block(block: Any, dtype: mx.Dtype, graph: str | None) -> Layer:
    """`CSPNeXtBlock` / `DarknetBottleneck`: conv2(conv1(x)), plus x with the identity shortcut."""
    path = chain(module(block.conv1, dtype, graph), module(block.conv2, dtype, graph))
    return (lambda x: x + path(x)) if block.identity else path


def _attention(att: Any) -> Layer:
    """Channel attention: x * hardsigmoid(fc(spatial mean)), the mean and the 1 x 1 conv float32."""
    w, b = arr(att.fc.weight[:, :, 0, 0].t(), mx.float32), arr(att.fc.bias, mx.float32)

    def apply(x: mx.array) -> mx.array:
        g = mx.mean(x.astype(mx.float32), axis=(1, 2), keepdims=True) @ w + b
        return x * mx.clip(g / 6 + 0.5, 0, 1).astype(x.dtype)

    return apply


def csp_layer(layer: Any, dtype: mx.Dtype, graph: str | None = None) -> Layer:
    """`CSPLayer` (CSPNeXt / RTMO): final(cat(blocks(main(x)), short(x))), CSPNeXt's channel attention before final."""
    main, short = conv_act(layer.main_conv, dtype, graph), conv_act(layer.short_conv, dtype, graph)
    blocks = chain(*(_block(b, dtype, graph) for b in layer.blocks))
    attention = _attention(layer.attention) if getattr(layer, "attention", None) is not None else None
    final = conv_act(layer.final_conv, dtype, graph)

    def apply(x: mx.array) -> mx.array:
        y = mx.concatenate([blocks(main(x)), short(x)], axis=-1)
        return final(attention(y) if attention is not None else y)

    return apply


def max_pool(x: mx.array) -> mx.array:
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


def spp(layer: Any, dtype: mx.Dtype, graph: str | None = None) -> Layer:
    """`SPP`: conv1, its 5 / 9 / 13 max pools concatenated, conv2."""
    first, second = conv_act(layer.conv1, dtype, graph), conv_act(layer.conv2, dtype, graph)

    def apply(x: mx.array) -> mx.array:
        maps = [first(x)]
        for _ in SPP_POOLS:  # 5, 9 = 5 twice, 13 = 5 three times
            maps.append(max_pool(maps[-1]))
        return second(mx.concatenate(maps, axis=-1))

    return apply


def _stage(stage: Any, dtype: mx.Dtype, graph: str | None) -> Layer:
    """The stride-2 conv, the SPP (last stage), the `CSPLayer`."""
    build = {"CSPLayer": csp_layer, "SPP": spp}
    return chain(*(build.get(base_type(layer).__name__, module)(layer, dtype, graph) for layer in stage))


def backbone(
    bb: Any, dtype: mx.Dtype, graph: str | None = None, *, step: bool = False
) -> Callable[[mx.array], tuple[mx.array, mx.array, mx.array]]:
    """`CSPNeXt`: NHWC `x` in `dtype` -> the stride 8, 16 and 32 maps; `step`: each stage a `stepped` piece."""
    stem = chain(*(module(m, dtype, graph) for m in bb.stem))
    stages = [_stage(getattr(bb, f"stage{i}"), dtype, graph) for i in range(1, 5)]
    piece = stepped if step else (lambda f: f)
    first = piece(lambda x: stages[1](stages[0](stem(x))))
    s3, s4 = piece(stages[2]), piece(stages[3])

    def apply(x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        c3 = first(x)
        c4 = s3(c3)
        return c3, c4, s4(c4)

    return apply


def up2(x: mx.array) -> mx.array:
    """Nearest x2 upsampling of an NHWC map."""
    n, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None], (n, h, 2, w, 2, c)).reshape(n, 2 * h, 2 * w, c)


def neck(
    layer: Any, dtype: mx.Dtype, graph: str | None = None, *, step: bool = False
) -> Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]]:
    """`CSPNeXtPAFPN` without its output convs: (stride 8, 16, 32 maps) -> (the three bottom-up maps).

    `step`: each top-down and bottom-up block a `stepped` piece.
    """
    reduce = [conv_act(m, dtype, graph) for m in layer.reduce_layers]
    down = [module(m, dtype, graph) for m in layer.downsamples]
    top = [csp_layer(m, dtype, graph) for m in layer.top_down_blocks]
    bottom = [csp_layer(m, dtype, graph) for m in layer.bottom_up_blocks]
    piece = stepped if step else (lambda f: f)

    def _top4(f4: mx.array, f5: mx.array) -> tuple[mx.array, mx.array]:
        t5 = reduce[0](f5)
        return t5, reduce[1](top[0](mx.concatenate([up2(t5), f4], axis=-1)))

    top4 = piece(_top4)
    top3 = piece(lambda f3, t4: top[1](mx.concatenate([up2(t4), f3], axis=-1)))
    bottom4 = piece(lambda m3, t4: bottom[0](mx.concatenate([down[0](m3), t4], axis=-1)))
    bottom5 = piece(lambda o4, t5: bottom[1](mx.concatenate([down[1](o4), t5], axis=-1)))

    def apply(f3: mx.array, f4: mx.array, f5: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        t5, t4 = top4(f4, f5)
        m3 = top3(f3, t4)
        o4 = bottom4(m3, t4)
        return m3, o4, bottom5(o4, t5)

    return apply
