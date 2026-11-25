"""EfficientViT-SAM's image encoder on Apple MLX, written in `mlx.core` operations (`MlxEncoder(trunk, norm, dtype)`).

`Trunk` (the large backbone and the SAM neck) plus the final `LayerNorm2d`, channels last, without an exported graph.
The eval BatchNorms are folded into the convolutions (float64 on the host). The 1x1 convolutions are matrix products,
the depthwise ones `graph_mlx`'s kernel; in float16 the dense 3 x 3 convolutions MLX would run as its own Winograd
transform (17x the error of the direct convolution) run as `conv_mlx`'s F(2,3) Winograd, the strided ones as im2col
and one GEMM. The ReLU linear attention (sums over every position), the neck's bicubic upsampling and the final norm
are float32 as upstream. Each stage, and the neck, is one `mx.compile`d function; a stage is evaluated while the next
is scheduled.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import mx
from fastforks.kernels.conv_mlx import ops as conv_mlx
from fastforks.kernels.graph_mlx.ops import depthwise
from fastforks.shared.mlx_graph import _mlx_winograd

from .net.ops import ConvLayer, EfficientViTBlock, FusedMBConv, LiteMLA, MBConv, ResBlock, ResidualBlock

if TYPE_CHECKING:
    from collections.abc import Callable

    from .fast import Trunk

__all__ = ["MlxEncoder"]

WINO_M = 2  # F(2,3) (relative error ~7e-4)
GELU_C = math.sqrt(2 / math.pi)


def _gelu(x: mx.array) -> mx.array:
    """`nn.GELU(approximate="tanh")`."""
    return 0.5 * x * (1 + mx.tanh(GELU_C * (x + 0.044715 * x * x * x)))


ACTS: dict[type, Callable[[mx.array], mx.array]] = {
    torch.nn.GELU: _gelu,
    torch.nn.ReLU: lambda x: mx.maximum(x, 0),
    torch.nn.Hardswish: lambda x: x * mx.clip(x + 3, 0, 6) / 6,
}


def _arr(a: np.ndarray, dtype: mx.Dtype) -> mx.array:
    out = mx.array(np.ascontiguousarray(a, dtype=np.float32)).astype(dtype)
    mx.eval(out)
    return out


def _folded(conv: torch.nn.Conv2d, bn: Any) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - a BatchNorm2d or None
    """`conv` followed by the eval `bn` (if any): weight `[O, I/g, kh, kw]` and bias `[O]` (float64)."""
    w = conv.weight.detach().double().numpy()
    b = conv.bias.detach().double().numpy() if conv.bias is not None else np.zeros(w.shape[0])
    if not isinstance(bn, torch.nn.BatchNorm2d):
        return w, b
    s = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy() + bn.eps)
    mean, shift = bn.running_mean.detach().double().numpy(), bn.bias.detach().double().numpy()
    return w * s[:, None, None, None], (b - mean) * s + shift


class _Conv:
    """`ConvLayer` (or a bare `nn.Conv2d`): 1x1 as a matrix product, depthwise, or dense k x k; bias and activation."""

    def __init__(self, layer: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - a `ConvLayer` or an `nn.Conv2d`
        conv = layer.conv if isinstance(layer, ConvLayer) else layer
        norm = layer.norm if isinstance(layer, ConvLayer) else None
        act = layer.act if isinstance(layer, ConvLayer) else None
        if isinstance(act, torch.nn.GELU) and act.approximate != "tanh":
            msg = "exact GELU"
            raise NotImplementedError(msg)
        self.act = None if act is None or isinstance(act, torch.nn.Identity) else ACTS[type(act)]
        w, b = _folded(conv, norm)
        self.stride, self.pad, self.groups = int(conv.stride[0]), int(conv.padding[0]), int(conv.groups)
        self.k, self.cout = w.shape[2], w.shape[0]
        self.b = _arr(b, dtype)
        self.u = None
        self.cols: mx.array | None = None
        if self.groups == self.cout and w.shape[1] == 1:  # depthwise
            self.kind = "dw"
            self.w = _arr(w[:, 0].transpose(1, 2, 0), dtype)  # [kh, kw, C]
        elif self.k == 1 and self.groups == 1:
            self.kind = "mm"
            self.w = _arr(w[:, :, 0, 0].T, dtype)  # [I, O]
        elif self.k == 1:  # grouped 1x1 (the attention's aggregation): [G, I/G, O/G]
            self.kind = "gmm"
            g = self.groups
            self.w = _arr(w[:, :, 0, 0].reshape(g, self.cout // g, -1).transpose(0, 2, 1), dtype)
        else:
            if self.groups != 1:
                msg = f"grouped {self.k} x {self.k} convolution"
                raise NotImplementedError(msg)
            self.kind = "dense"
            ohwi = w.transpose(0, 2, 3, 1)
            self.w = _arr(ohwi, dtype)
            if dtype == mx.float16 and self.k == 3 and self.stride == 1:  # noqa: PLR2004 - conv_mlx is 3 x 3 only
                self.u = conv_mlx.weights(mx.array(ohwi.astype(np.float32)), WINO_M, dtype)
            if self.stride > 1:  # strided: im2col and one GEMM, 20-25 % faster than `mx.conv2d` (rows `(i, j, c)`)
                self.cols = _arr(w.transpose(2, 3, 1, 0).reshape(-1, self.cout), dtype)

    def _dense(self, x: mx.array) -> mx.array:
        if self.u is not None and conv_mlx.supported(x.shape, self.cout):
            return conv_mlx.conv3x3(x, self.u, self.b, m=WINO_M)
        if self.cols is not None:
            k, st, p = self.k, self.stride, self.pad
            ho, wo = (x.shape[1] + 2 * p - k) // st + 1, (x.shape[2] + 2 * p - k) // st + 1
            xp = mx.pad(x, [(0, 0), (p, p), (p, p), (0, 0)])
            span_h, span_w = st * (ho - 1) + 1, st * (wo - 1) + 1
            taps = [xp[:, i : i + span_h : st, j : j + span_w : st] for i in range(k) for j in range(k)]
            return mx.addmm(self.b, mx.concatenate(taps, axis=-1), self.cols)
        positions = (x.shape[1] // self.stride) * (x.shape[2] // self.stride)
        s = (self.stride, self.stride)
        if x.dtype == mx.float16 and _mlx_winograd(self.w.shape, self.cout, s, (1, 1), 1, positions):
            y = mx.conv2d(x.astype(mx.float32), self.w.astype(mx.float32), stride=self.stride, padding=self.pad)
            return (y + self.b.astype(mx.float32)).astype(x.dtype)
        return mx.conv2d(x, self.w, stride=self.stride, padding=self.pad) + self.b

    def __call__(self, x: mx.array) -> mx.array:
        if self.kind == "mm":
            y = x @ self.w + self.b
        elif self.kind == "dw":
            s, p = (self.stride, self.stride), (self.pad, self.pad)
            y = depthwise(x, self.w, self.b, s, p, (1, 1))
        elif self.kind == "gmm":
            n, h, w, c = x.shape
            g = self.w.shape[0]
            y = (x.reshape(h * w, g, c // g).transpose(1, 0, 2) @ self.w).transpose(1, 0, 2)
            y = y.reshape(n, h, w, self.cout) + self.b
        else:
            y = self._dense(x)
        return y if self.act is None else self.act(y)


class _Seq:
    """Convolutions in a row (`MBConv`, `FusedMBConv`, `ResBlock`)."""

    def __init__(self, convs: list[Any], dtype: mx.Dtype) -> None:
        self.convs = [_Conv(c, dtype) for c in convs]

    def __call__(self, x: mx.array) -> mx.array:
        for c in self.convs:
            x = c(x)
        return x


class _Attention:
    """`LiteMLA`: multi-scale q / k / v, ReLU linear attention in float32, projection."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - a `LiteMLA`
        self.dim, self.eps = m.dim, float(m.eps)
        self.qkv = _Conv(m.qkv, dtype)
        self.aggreg = [(_Conv(seq[0], dtype), _Conv(seq[1], dtype)) for seq in m.aggreg]
        self.proj = _Conv(m.proj, dtype)

    def __call__(self, x: mx.array) -> mx.array:
        qkv = self.qkv(x)
        qkv = mx.concatenate([qkv, *(pw(dw(qkv)) for dw, pw in self.aggreg)], axis=-1)
        n, h, w, c = qkv.shape
        d = self.dim
        t = qkv.astype(mx.float32).reshape(h * w, c // (3 * d), 3 * d).transpose(1, 0, 2)  # [G, HW, 3d]
        q, k, v = mx.maximum(t[..., :d], 0), mx.maximum(t[..., d : 2 * d], 0), t[..., 2 * d :]
        kv = k.transpose(0, 2, 1) @ v  # [G, d, d]
        num = q @ kv  # [G, HW, d]
        den = q @ k.sum(axis=1)[..., None]  # [G, HW, 1]
        out = (num / (den + self.eps)).transpose(1, 0, 2).reshape(n, h, w, -1)
        return self.proj(out.astype(x.dtype))


def _block(m: Any, dtype: mx.Dtype) -> Callable[[mx.array], mx.array]:  # noqa: ANN401 - a block of several types
    """A backbone / neck block: a `ResidualBlock` (shortcut or none) or an `EfficientViTBlock`."""
    if isinstance(m, EfficientViTBlock):
        ctx, local = _block(m.context_module, dtype), _block(m.local_module, dtype)
        return lambda x: local(ctx(x))
    if isinstance(m, ConvLayer):
        return _Conv(m, dtype)
    if not isinstance(m, ResidualBlock) or m.pre_norm is not None or m.post_act:
        msg = f"unsupported block {type(m).__name__}"
        raise TypeError(msg)
    main = m.main
    if isinstance(main, LiteMLA):
        f: Callable[[mx.array], mx.array] = _Attention(main, dtype)
    elif isinstance(main, MBConv):
        f = _Seq([main.inverted_conv, main.depth_conv, main.point_conv], dtype)
    elif isinstance(main, FusedMBConv):
        f = _Seq([main.spatial_conv, main.point_conv], dtype)
    elif isinstance(main, ResBlock):
        f = _Seq([main.conv1, main.conv2], dtype)
    else:
        msg = f"unsupported block {type(main).__name__}"
        raise TypeError(msg)
    return f if m.shortcut is None else lambda x: f(x) + x


def _chain(ops: list[Callable[[mx.array], mx.array]]) -> Callable[[mx.array], mx.array]:
    """The blocks of a stage applied in turn."""

    def run(x: mx.array) -> mx.array:
        for op in ops:
            x = op(x)
        return x

    return run


def _bicubic(n_in: int, n_out: int) -> mx.array:
    """`F.interpolate(mode="bicubic", align_corners=False)` along one axis as a float32 `[n_out, n_in]` matrix."""
    real = (n_in / n_out) * (np.arange(n_out) + 0.5) - 0.5
    i0 = np.floor(real).astype(np.int64)
    t = real - i0
    a = -0.75
    weights = [
        ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
        ((a + 2) * t - (a + 3)) * t * t + 1,
        ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
        ((a * (2 - t) - 5 * a) * (2 - t) + 8 * a) * (2 - t) - 4 * a,
    ]
    m = np.zeros((n_out, n_in))
    for k, wk in enumerate(weights):
        np.add.at(m, (np.arange(n_out), np.clip(i0 + k - 1, 0, n_in - 1)), wk)
    return mx.array(m.astype(np.float32))


class MlxEncoder:
    """`Trunk` and the final `LayerNorm2d` of `fast.Trunk` (eval, float32, on the CPU) on MLX in `dtype` (the name).

    `[1, S, S, 3]` float32 normalised input -> the float32 image embedding `[1, 64, 64, 256]`.
    """

    def __init__(self, trunk: Trunk, norm: Any, dtype: str) -> None:  # noqa: ANN401 - the `LayerNorm2d`
        """Copy the weights of `trunk` and `norm` to MLX."""
        self.dtype = getattr(mx, dtype)
        backbone: Any = trunk.backbone
        neck: Any = trunk.neck
        self.stages = [[_block(op, self.dtype) for op in stage.op_list] for stage in backbone.stages]
        self.inputs = []
        for fid, seq in zip(neck.input_keys, neck.input_ops, strict=True):
            self.inputs.append((int(fid.removeprefix("stage")), _Conv(seq.op_list[0], self.dtype)))
        self.middle = [_block(op, self.dtype) for op in neck.middle.op_list]
        self.out = _Conv(neck.output_ops[0].op_list[0], self.dtype)
        self.norm_w = mx.array(norm.weight.detach().float().numpy())
        self.norm_b = mx.array(norm.bias.detach().float().numpy())
        self.norm_eps = float(norm.eps)
        self.side = 64  # SAM's image embedding
        self.mats: dict[int, mx.array] = {}
        # one `mx.compile`d function per stage (the weights its constants): fuses the bias / activation / residual
        # element-wise work around the convolutions, ~40 % of the uncompiled stage time
        self.stage_fns = [mx.compile(_chain(stage)) for stage in self.stages]
        self.neck_fn = mx.compile(self._neck)

    def _up(self, x: mx.array) -> mx.array:
        """The neck's `UpSampleLayer(size=(64, 64))`: bicubic in float32 (a square input: one matrix per side)."""
        n = x.shape[1]
        if n == self.side:
            return x
        if n not in self.mats:
            self.mats[n] = _bicubic(n, self.side)
        m = self.mats[n]
        x = x.astype(mx.float32)
        y = mx.einsum("oh,nhwc->nowc", m, x)
        return mx.einsum("pw,nowc->nopc", m, y)

    def __call__(self, x: mx.array) -> mx.array:
        """The image embedding of the normalised `[1, S, S, 3]` input; one stage evaluated at a time."""
        x = x.astype(self.dtype)
        feats, pending = [], None
        for fn in self.stage_fns:
            x = fn(x)
            mx.async_eval(x)
            if pending is not None:
                mx.eval(pending)
            feats.append(x)
            pending = x
        return self.neck_fn(*(feats[i] for i in sorted({i for i, _ in self.inputs})))

    def _neck(self, *feats: mx.array) -> mx.array:
        """The SAM neck and the final norm of the stages' features (the `inputs` stages, ascending)."""
        by_stage = dict(zip(sorted({i for i, _ in self.inputs}), feats, strict=True))
        acc = None
        for i, conv in self.inputs:
            y = self._up(conv(by_stage[i]))
            acc = y if acc is None else acc + y
        assert acc is not None  # noqa: S101 - the neck has inputs
        x = acc.astype(self.dtype)
        for op in self.middle:
            x = op(x)
        y = self.out(x).astype(mx.float32)
        y = y - y.mean(axis=-1, keepdims=True)
        y = y * mx.rsqrt(mx.square(y).mean(axis=-1, keepdims=True) + self.norm_eps)
        return y * self.norm_w + self.norm_b
