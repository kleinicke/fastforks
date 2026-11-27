"""Building blocks of the hand-written MLX optical-flow networks: state-dict weights, convs, pyramids, upsampling.

For a family's `mlx_net.py` built by `mlx_port.recorded` from the checkpoint's state dict (a dict of empty tensors of
the safetensors header on a conversion-cache hit, `mlx_port.header_skeleton`): every weight is read through `put`,
which converts in numpy (float64) on a miss and only reads shapes on a hit, so a hit runs no torch operation (torch
arithmetic on a meta tensor imports `torch._dynamo`).

- `conv` / `depthwise` / `linear`: a `Conv2d` / `Linear` (an eval `BatchNorm2d` folded, output channels selected,
  scaled) applied to NHWC maps: 1 x 1 convs, patch convs (kernel = stride) and linears as GEMMs, depthwise convs
  through `graph_mlx.ops.depthwise`, the others through the graph translator's tuned dispatch (`mlx_graph._conv2d`:
  channel padding onto MLX's fast tiles, half-precision Winograd convs in float32 when the network sets
  `half_wino_f32`, see `options`);
- `pyramid`: the 2 x 2 average-pooled levels of a feature map for `graph_mlx.ops.corr_windows` (upstream halves
  bilinearly with align_corners=False, which is the same pooling; pooling commutes with the dot product, so the
  windows need no all-pairs volume);
- `ConvNext` / `refine`: SEA-RAFT's update-block refinement (`ConvNextBlock`s on `[hidden | inputs]`), shared by
  MEMFOF and ReCoVEr;
- `eager`: a compiled step evaluated at once, the step in flight awaited first (the full-resolution encoder convs:
  MLX frees a command buffer's transients only when it completes, so two pipelined Winograd convs at 1/2 resolution
  hold both convs' transforms, MEMFOF at 1080 x 1920 float16 +1.9 GB);
- `grid`: pixel coordinates; `convex_rows` / `convex_upsample`: RAFT's convex upsampling with the mask head's
  outputs reordered so the softmax runs over the last axis and the combination is one batched matrix product.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared import mlx_port
from fastforks.shared.cspnext import mlx_net as cspnext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "DTYPES",
    "Conv",
    "ConvNext",
    "Depthwise",
    "blocks",
    "conv",
    "convex_rows",
    "convex_upsample",
    "depthwise",
    "eager",
    "gelu",
    "grid",
    "linear",
    "np64",
    "options",
    "put",
    "pyramid",
    "refine",
    "relu",
]

DTYPES = {"float32": mx.float32, "float16": mx.float16}
"""Run dtype names (the `ctx.dtype` strings) -> MLX dtypes."""


def np64(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    """A (real) torch tensor as a float64 numpy array: only called on a cache miss."""
    return t.detach().double().numpy()


def put(shape: Sequence[int], dtype: mx.Dtype, make: Callable[[], np.ndarray]) -> mx.array:
    """The weight `make()` (float64 numpy, only called on a miss) as an evaluated MLX array in `dtype`, cached.

    `shape` is what the replay checks the cached array against (computed from the skeleton's shapes).
    """

    def convert() -> mx.array:
        a = np.asarray(make())
        if a.shape != tuple(shape):
            msg = f"MLX flow port: converted weight {a.shape}, expected {tuple(shape)}"
            raise ValueError(msg)
        out = mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype)
        mx.eval(out)
        return out

    return mlx_port.tensor(SimpleNamespace(shape=tuple(int(n) for n in shape)), dtype, convert)


def relu(x: mx.array) -> mx.array:
    """ReLU."""
    return mx.maximum(x, 0)


def options(*, half_wino_f32: bool) -> None:
    """Set the translator's module options `Conv` reads for a call (portable mode, 1 x 1 GEMMs)."""
    mg._KERNELS, mg._GEMM1X1 = False, True  # noqa: SLF001
    mg._HALF_WINO_F32, mg._HALF_WINO_PAD = half_wino_f32, False  # noqa: SLF001


class Conv:
    """A dense convolution of NHWC maps: OHWI weight, bias, stride, padding.

    Two forms run as GEMMs: a 1 x 1 unpadded conv (`[out, in]` weight, strided by subsampling) and a patch conv
    (kernel = stride, unpadded: `[out, (kh, kw, in)]` weight over the non-overlapping patches).
    """

    def __init__(self, w: mx.array, b: mx.array, stride: int, padding: int, *, patch: int = 0) -> None:
        """Hold the converted weights (`patch`: the patch conv's kernel size)."""
        self.w, self.b, self.stride, self.padding, self.patch = w, b, stride, padding, patch
        self.gemm = w.ndim == 2  # noqa: PLR2004 - `[out, in]`

    @property
    def out(self) -> int:
        """Output channels."""
        return int(self.w.shape[0])

    def __call__(self, x: mx.array) -> mx.array:
        """The conv (bias added) of `[N, H, W, C]`."""
        if self.patch:
            n, h, w, c = x.shape
            k = self.patch
            h, w = h // k, w // k
            p = x[:, : h * k, : w * k].reshape(n, h, k, w, k, c).transpose(0, 1, 3, 2, 4, 5).reshape(-1, k * k * c)
            return mx.addmm(self.b, p, self.w.T).reshape(n, h, w, -1)
        if self.gemm:
            if self.stride > 1:
                x = x[:, :: self.stride, :: self.stride]
            return mg.conv1x1(x, self.w.T, self.b)
        s, p = (self.stride, self.stride), (self.padding, self.padding)
        return mg._conv2d(x, self.w, s, p, (1, 1), 1) + self.b  # noqa: SLF001 - the translator's tuned dispatch


class Depthwise:
    """A depthwise convolution of NHWC maps (`graph_mlx.ops.depthwise`): `[kh, kw, C]` weight, bias, stride."""

    def __init__(self, w: mx.array, b: mx.array, stride: int, padding: int) -> None:
        """Hold the converted weights."""
        self.w, self.b, self.stride, self.padding = w, b, stride, padding

    def __call__(self, x: mx.array) -> mx.array:
        """The conv (bias added) of `[N, H, W, C]`."""
        s, p = (self.stride, self.stride), (self.padding, self.padding)
        return ops.depthwise(x, self.w, self.b, s, p, (1, 1))


def _folded(sd: Mapping[str, Any], name: str, bn: str | None, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """`name`'s OIHW weight and bias (float64), the eval batch norm `bn` (`eps`) folded in."""
    w = np64(sd[f"{name}.weight"])
    b = np64(sd[f"{name}.bias"]) if f"{name}.bias" in sd else np.zeros(w.shape[0])
    if bn is not None:
        s = np64(sd[f"{bn}.weight"]) / np.sqrt(np64(sd[f"{bn}.running_var"]) + eps)
        w = w * s[:, None, None, None]
        b = np64(sd[f"{bn}.bias"]) + (b - np64(sd[f"{bn}.running_mean"])) * s
    return w, b


def conv(  # noqa: PLR0913 - the conv's own arguments and its rewrites
    sd: Mapping[str, Any],
    name: str,
    dtype: mx.Dtype,
    *,
    stride: int = 1,
    padding: int | None = None,
    bn: str | None = None,
    eps: float = 1e-5,
    rows: Sequence[int] | None = None,
    scale: float = 1.0,
    inputs: Callable[[np.ndarray], np.ndarray] | None = None,
    cin: int | None = None,
    bias: bool = True,
) -> Conv:
    """The `Conv2d` `name` of the state dict `sd` in `dtype`.

    Args:
        sd: The state dict (real tensors on a miss, empty ones on a hit).
        name: The conv's key prefix.
        dtype: The weights' dtype.
        stride: The conv's stride.
        padding: Its padding (default `kernel // 2`).
        bn: Key prefix of an eval `BatchNorm2d` after it, folded in.
        eps: The batch norm's epsilon.
        rows: Output channels kept, in this order.
        scale: Factor on the weight and bias.
        inputs: Rewrites the OIHW weight's input channels (e.g. merging duplicated inputs); `cin` is the result's count.
        cin: Input channels after `inputs`.
        bias: False drops the bias (zeros).
    """
    o, i, kh, kw = (int(n) for n in sd[f"{name}.weight"].shape)
    out = o if rows is None else len(rows)
    i = i if cin is None else cin
    pad = kh // 2 if padding is None else padding

    def weights() -> tuple[np.ndarray, np.ndarray]:
        w, b = _folded(sd, name, bn, eps)
        if not bias:
            b = np.zeros_like(b)
        if inputs is not None:
            w = inputs(w)
        if rows is not None:
            w, b = w[list(rows)], b[list(rows)]
        return w * scale, b * scale

    cache: list[tuple[np.ndarray, np.ndarray]] = []

    def get() -> tuple[np.ndarray, np.ndarray]:
        if not cache:
            cache.append(weights())
        return cache[0]

    if kh == kw == 1 and pad == 0:
        w = put((out, i), dtype, lambda: get()[0][:, :, 0, 0])
    elif kh == kw == stride and pad == 0:
        w = put((out, kh * kw * i), dtype, lambda: get()[0].transpose(0, 2, 3, 1).reshape(out, kh * kw * i))
        return Conv(w, put((out,), dtype, lambda: get()[1]), stride, pad, patch=kh)
    else:
        w = put((out, kh, kw, i), dtype, lambda: get()[0].transpose(0, 2, 3, 1))
    return Conv(w, put((out,), dtype, lambda: get()[1]), stride, pad)


def depthwise(  # noqa: PLR0913 - the conv's own arguments
    sd: Mapping[str, Any],
    name: str,
    dtype: mx.Dtype,
    *,
    stride: int = 1,
    bn: str | None = None,
    eps: float = 1e-5,
) -> Depthwise:
    """The depthwise `Conv2d` `name` of `sd` in `dtype` (padding `kernel // 2`, the batch norm `bn` folded in)."""
    c, _, kh, kw = (int(n) for n in sd[f"{name}.weight"].shape)
    cache: list[tuple[np.ndarray, np.ndarray]] = []

    def get() -> tuple[np.ndarray, np.ndarray]:
        if not cache:
            cache.append(_folded(sd, name, bn, eps))
        return cache[0]

    w = put((kh, kw, c), dtype, lambda: get()[0][:, 0].transpose(1, 2, 0))
    return Depthwise(w, put((c,), dtype, lambda: get()[1]), stride, kh // 2)


def linear(sd: Mapping[str, Any], name: str, dtype: mx.Dtype, *, scale: Any = None) -> tuple[mx.array, mx.array]:  # noqa: ANN401 - a tensor key
    """The `Linear` `name` as (`[out, in]` weight, bias); `scale`: the key of a per-output factor folded in."""
    o, i = (int(n) for n in sd[f"{name}.weight"].shape)

    def factor() -> np.ndarray:
        return np.ones(o) if scale is None else np64(sd[scale]).reshape(o)

    w = put((o, i), dtype, lambda: np64(sd[f"{name}.weight"]) * factor()[:, None])
    b = put((o,), dtype, lambda: np64(sd[f"{name}.bias"]) * factor())
    return w, b


def gelu(x: mx.array) -> mx.array:
    """Exact GELU (erf), as `nn.GELU()`."""
    return 0.5 * x * (1 + mx.erf(x * (1 / math.sqrt(2))))


class ConvNext:
    """SEA-RAFT's `ConvNextBlock`, its layer scale `gamma` folded into `pwconv2`.

    7 x 7 depthwise conv, LayerNorm (eps 1e-6), Linear-GELU-Linear, layer scale, residual, then the 1 x 1 `final`.
    """

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        """Convert the block `p` of `sd`."""
        c = int(sd[f"{p}.dwconv.weight"].shape[0])
        self.dw = depthwise(sd, f"{p}.dwconv", dtype)
        self.lw = put((c,), dtype, lambda: np64(sd[f"{p}.norm.weight"]))
        self.lb = put((c,), dtype, lambda: np64(sd[f"{p}.norm.bias"]))
        self.w1, self.b1 = linear(sd, f"{p}.pwconv1", dtype)
        self.w2, self.b2 = linear(sd, f"{p}.pwconv2", dtype, scale=f"{p}.gamma")
        self.final = conv(sd, f"{p}.final", dtype)

    def __call__(self, x: mx.array) -> mx.array:
        """`[1, h, w, C]` -> `[1, h, w, out]`."""
        _, h, w, _ = x.shape
        y = mx.fast.layer_norm(self.dw(x), self.lw, self.lb, 1e-6)
        y = gelu(mx.addmm(self.b1, y.reshape(h * w, -1), self.w1.T))
        y = mx.addmm(self.b2, y, self.w2.T).reshape(x.shape)
        return self.final(x + y)


def refine(blocks: Sequence[ConvNext], net: mx.array, inputs: Sequence[mx.array]) -> mx.array:
    """The update block's refinement: `net = block([net | inputs])` for each block."""
    for block in blocks:
        net = block(mx.concatenate([net, *inputs], axis=-1))
    return net


def blocks(sd: Mapping[str, Any], prefix: str, dtype: mx.Dtype) -> list[ConvNext]:
    """The `ConvNext` blocks `prefix.0`, `prefix.1`, ... of `sd`."""
    out = []
    while f"{prefix}.{len(out)}.dwconv.weight" in sd:
        out.append(ConvNext(sd, f"{prefix}.{len(out)}", dtype))
    return out


def eager(f: Callable[..., Any]) -> Callable[..., Any]:
    """`f` compiled, each call evaluated before it returns, after the `cspnext.stepped` step in flight."""
    g = mx.compile(f)

    def run(*args: Any) -> Any:  # noqa: ANN401 - the step's arrays
        if cspnext._PENDING:  # noqa: SLF001 - the shared pipeline of `stepped`
            mx.eval(cspnext._PENDING.pop())  # noqa: SLF001
        y = g(*args)
        mx.eval(y)
        return y

    return run


def _halve(x: mx.array) -> mx.array:
    """`F.avg_pool2d(x, 2)` of an NHWC map (an odd last row / column dropped), summed in float32."""
    n, h, w, c = x.shape
    h, w = h // 2, w // 2
    y = x[:, : 2 * h, : 2 * w].astype(mx.float32).reshape(n, h, 2, w, 2, c).sum(axis=(2, 4))
    return (0.25 * y).astype(x.dtype)


def pyramid(f: mx.array, levels: int) -> list[mx.array]:
    """`f` and its `levels - 1` 2 x 2 average-pooled halvings (in `f`'s dtype)."""
    out = [f]
    for _ in range(levels - 1):
        out.append(_halve(out[-1]))
    return out


def grid(h: int, w: int) -> mx.array:
    """Pixel coordinates `[1, h, w, 2]` (x, y) float32."""
    xs = mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None], (h, w))
    ys = mx.broadcast_to(mx.arange(h, dtype=mx.float32)[:, None], (h, w))
    return mx.stack([xs, ys], axis=-1)[None]


def convex_rows(factor: int, offset: int = 0) -> list[int]:
    """Mask-head outputs `offset + (k, i, j)` (upstream's `view(9, F, F)`) in `(i, j, k)` order (`convex_upsample`)."""
    return [
        offset + k * factor * factor + i * factor + j for i in range(factor) for j in range(factor) for k in range(9)
    ]


def convex_upsample(flow: mx.array, mask: mx.array, factor: int) -> mx.array:
    """RAFT's convex upsampling of `flow` `[1, h, w, 2]` by `factor`: float32 `[1, 2, F h, F w]`.

    `mask` is the head's `[1, h, w, F * F * 9]` logits in `convex_rows` order: softmax over the 9 neighbours (float32),
    each output pixel the weighted sum of `F * flow` over its coarse pixel's 3 x 3 neighbourhood (zeros outside).
    """
    _, h, w, _ = flow.shape
    m = mx.softmax(mask.reshape(h * w, factor * factor, 9).astype(mx.float32), axis=-1, precise=True)
    fp = mx.pad(factor * flow[0].astype(mx.float32), ((1, 1), (1, 1), (0, 0)))
    near = mx.stack([fp[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=2)  # [h, w, 9, 2]
    up = (m @ near.reshape(h * w, 9, 2)).reshape(h, w, factor, factor, 2)
    return up.transpose(4, 0, 2, 1, 3).reshape(1, 2, factor * h, factor * w)
