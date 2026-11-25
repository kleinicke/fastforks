"""RealPLKSR on Apple MLX, written in `mlx.core` operations (`MlxRealPLKSR(net, dtype)`): image -> upscaled image.

`realplksr.forward` of the vendored upstream network (eval path), channels last. Written out instead of replayed from
an exported graph: the export, translation and segment tuning were 1.5-2 s of a 2.7-3.9 s first call at 512 x 512.

- the dense 3 x 3 convs (channel mixer, element-wise attention, the output conv with the repeated input as its
  residual) are `conv_mlx`'s Winograd F(4,3) when it supports the shape, else `mx.conv2d`; Mish one Metal kernel;
- the 17 x 17 conv of the first channels is `mlx_ops.rows_conv` (row GEMMs, 2.2x `mx.conv2d`);
- GroupNorm's statistics are float32 per-channel moments (`graph_mlx.ops.channel_stats`, shifted, one pass) combined
  per group, applied with the skip as one per-channel scale and shift;
- the pixel shuffle a reshape.

Each block is evaluated as it is scheduled, one in flight (`_step`): MLX frees a command buffer's transients only
when it completes, and one lazy graph of the whole network holds far more than its live set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.conv_mlx import ops as conv
from fastforks.kernels.graph_mlx import ops

from .mlx_ops import rows_conv

__all__ = ["MlxRealPLKSR"]


def _np(t: Any) -> np.ndarray:  # noqa: ANN401 - a torch tensor
    return t.detach().float().numpy()


class _Conv3:
    """A 3 x 3 stride-1 conv (zero padding 1): Winograd F(4,3) where `conv_mlx` supports the shape, else `mx.conv2d`."""

    def __init__(self, m: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - a `Conv2d`
        self.w = mx.array(np.ascontiguousarray(_np(m.weight).transpose(0, 2, 3, 1)))  # OHWI float32
        self.b = mx.array(_np(m.bias)).astype(dtype)
        self.dtype = dtype
        self.u: mx.array | None = None
        self.co = m.out_channels
        self.ok: dict[tuple[int, ...], bool] = {}

    def __call__(self, x: mx.array, residual: mx.array | None = None) -> mx.array:
        if self.ok.setdefault(x.shape, conv.supported(x.shape, self.co)):
            if self.u is None:
                self.u = conv.weights(self.w, 4, self.dtype)
            return conv.conv3x3(x, self.u, self.b, m=4, residual=residual)
        y = mx.conv2d(x, self._direct(), padding=1) + self.b
        return y if residual is None else y + residual

    def _direct(self) -> mx.array:
        if self.w.dtype != self.dtype:
            self.w = self.w.astype(self.dtype)
            mx.eval(self.w)
        return self.w


@mx.compile
def _attend(x: mx.array, a: mx.array) -> mx.array:
    """Element-wise attention: `x * sigmoid(a)`."""
    return x * mx.sigmoid(a)


@mx.compile
def _norm_skip(y: mx.array, scale: mx.array, shift: mx.array, skip: mx.array) -> mx.array:
    """GroupNorm as a per-channel scale and shift (float32), plus the block's input."""
    return (y.astype(mx.float32) * scale + shift).astype(y.dtype) + skip


_PENDING: list[mx.array] = []  # the step in flight (`_step`)


def _step(x: mx.array) -> mx.array:
    """Schedule `x` and wait for the step before it (pipeline depth 1: the GPU stays busy, the transients bounded)."""
    mx.async_eval(x)
    if _PENDING:
        mx.eval(_PENDING.pop())
    _PENDING.append(x)
    return x


class _Block:
    """`PLKBlock.forward`."""

    def __init__(self, b: Any, dtype: mx.Dtype) -> None:  # noqa: ANN401 - a `PLKBlock`
        def arr(t: Any) -> mx.array:  # noqa: ANN401 - a torch tensor
            return mx.array(np.ascontiguousarray(_np(t))).astype(dtype)

        self.mix = (_Conv3(b.channel_mixer[0], dtype), _Conv3(b.channel_mixer[2], dtype))
        self.lk_w, self.lk_b = arr(b.lk.conv.weight), arr(b.lk.conv.bias)
        self.ea = _Conv3(b.attn.f[0], dtype) if hasattr(b.attn, "f") else None  # `use_ea=False`: Identity
        self.refine_w, self.refine_b = arr(b.refine.weight.permute(0, 2, 3, 1)), arr(b.refine.bias)
        self.groups, self.eps = b.norm.num_groups, b.norm.eps
        self.norm_w, self.norm_b = mx.array(_np(b.norm.weight)), mx.array(_np(b.norm.bias))
        mx.eval(self.lk_w, self.lk_b, self.refine_w, self.refine_b, self.norm_w, self.norm_b)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.mix[1](ops.mish(self.mix[0](x)))
        c = self.lk_w.shape[0]
        y = mx.concatenate([rows_conv(y[..., :c], self.lk_w) + self.lk_b, y[..., c:]], axis=-1)
        if self.ea is not None:
            y = _attend(y, self.ea(y))
        y = mx.conv2d(y, self.refine_w) + self.refine_b
        return _norm_skip(y, *self._group_norm(y), x)

    def _group_norm(self, y: mx.array) -> tuple[mx.array, mx.array]:
        """Per-sample, per-channel `[B, 1, 1, C]` scale and shift of `GroupNorm(groups)` over `y` (float32)."""
        n, _, _, c = y.shape
        scales, shifts = [], []
        for i in range(n):
            mean, var = ops.channel_stats(y[i].reshape(-1, c))
            mean, var = mean.reshape(self.groups, -1), var.reshape(self.groups, -1)
            gm = mx.mean(mean, axis=1, keepdims=True)
            gv = mx.mean(var, axis=1, keepdims=True) + mx.mean(mx.square(mean - gm), axis=1, keepdims=True)
            s = mx.broadcast_to(mx.rsqrt(gv + self.eps), mean.shape).reshape(c) * self.norm_w
            scales.append(s)
            shifts.append(self.norm_b - mx.broadcast_to(gm, mean.shape).reshape(c) * s)
        return mx.stack(scales).reshape(n, 1, 1, c), mx.stack(shifts).reshape(n, 1, 1, c)


class MlxRealPLKSR:
    """The network: `[B, 3, H, W]` float32 -> `[B, 3, s * H, s * W]` float32 (`realplksr.forward`, not clamped)."""

    def __init__(self, net: Any, dtype: str) -> None:  # noqa: ANN401 - the vendored `realplksr` (eval, float32)
        """The weights of `net` in `dtype`."""
        self.dtype = getattr(mx, dtype)
        feats = list(net.feats)
        self.scale = net.upscale
        self.head = _Conv3(feats[0], self.dtype)
        self.tail = _Conv3(feats[-1], self.dtype)
        self.blocks = [_Block(b, self.dtype) for b in feats[1:-2]]

    def __call__(self, image: mx.array) -> mx.array:
        """`realplksr.forward`."""
        try:
            return self._forward(image)
        finally:
            _PENDING.clear()

    def _forward(self, image: mx.array) -> mx.array:
        x = mx.transpose(image, (0, 2, 3, 1)).astype(self.dtype)
        s = self.scale
        y = _step(self.head(x))
        for blk in self.blocks:
            y = _step(blk(y))
        y = self.tail(y, residual=mx.repeat(x, s * s, axis=-1))  # + repeat_interleave(x, s^2) along channels
        n, h, w, c = y.shape
        y = y.reshape(n, h, w, c // (s * s), s, s).transpose(0, 3, 1, 4, 2, 5)
        return y.reshape(n, c // (s * s), h * s, w * s).astype(mx.float32)
