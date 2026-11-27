"""NeuFlow v2 on Apple MLX, written in `mlx.core` operations: `MlxNeuFlow(sd, dtype, iters)(image1, image2)` -> flow.

The maths of `net.NeuFlow.forward`, channels last, built from the checkpoint's state dict (`shared.mlx_flow`):

- the backbone (both images as one batch, batch norms folded) and the 1/8 merges in float32, as the graph
  translation keeps them (`mlx_cast.float32_call`); the cross-attention, the refinements and the upsampling head in
  the run dtype; the final batch norm of the cross-attention an affine;
- the global matching in float32 (`_core.mlx.attention`, the grid padded to a fused head size: no score matrix);
- no correlation volume: `graph_mlx.ops.corr_windows` takes each 9 x 9 window's dot products from the features;
- each refinement one compiled step, one in flight; its first conv split by inputs: the context and the constant
  radius channel are convolved once per level (a conv is linear, zero padding included), each step convolves only
  [correlation | iteration context | flow];
- the 8 x 8 stride-8 conv of image 1 a GEMM over the patches; the convex upsampling in float32.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxNeuFlow"]

LAYOUT = "neuflow-v2-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__))

FACTOR = 8
FEATURE, CONTEXT, ITER_CONTEXT = 128, 64, 64
RADIUS = 4
ITERS_S16 = 1
_CORR = (2 * RADIUS + 1) ** 2
# the refinement's first conv inputs: [corr 81 | context 64 | iteration context 64 | flow 2 | radius 1]
_STEP_IN = [*range(_CORR), *range(_CORR + CONTEXT, _CORR + CONTEXT + ITER_CONTEXT + 2)]
_CONST_IN = [*range(_CORR, _CORR + CONTEXT), _CORR + CONTEXT + ITER_CONTEXT + 2]


def _leaky(x: mx.array) -> mx.array:
    return mx.maximum(x, 0.1 * x)


def _block(sd: Mapping[str, Any], p: str, dtype: mx.Dtype, *, stride: int, padding: int) -> Callable[..., mx.array]:
    """`ConvBlock` `p`: conv-bn-LeakyReLU, 3 x 3 conv-bn-LeakyReLU."""
    c1 = mf.conv(sd, f"{p}.conv1", dtype, stride=stride, padding=padding, bn=f"{p}.norm1")
    c2 = mf.conv(sd, f"{p}.conv2", dtype, bn=f"{p}.norm2")
    return lambda x: _leaky(c2(_leaky(c1(x))))


def _merge(sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> Callable[..., mx.array]:
    """`_merge` `p`: conv, GELU, conv-bn."""
    c1 = mf.conv(sd, f"{p}.0", dtype)
    c2 = mf.conv(sd, f"{p}.2", dtype, bn=f"{p}.3")
    return lambda x: c2(mf.gelu(c1(x)))


def _up2(x: mx.array) -> mx.array:
    """Nearest 2x upsampling of `[N, h, w, C]`."""
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


class _Layer:
    """`TransformerLayer`: single-head cross-attention, merge + LayerNorm, MLP on [source | message] + LayerNorm."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        dim = int(sd[f"{p}.q_proj.weight"].shape[0])
        self.scale = dim**-0.5
        self.qkv = mf.put(
            (3 * dim, dim), dtype, lambda: np.concatenate([mf.np64(sd[f"{p}.{n}_proj.weight"]) for n in "qkv"])
        )
        self.qkv_b = mf.put(
            (3 * dim,), dtype, lambda: np.concatenate([mf.np64(sd[f"{p}.{n}_proj.bias"]) for n in "qkv"])
        )
        self.merge = mf.linear(sd, f"{p}.merge", dtype)
        self.n1 = (mf.put((dim,), dtype, lambda: mf.np64(sd[f"{p}.norm1.weight"])),
                   mf.put((dim,), dtype, lambda: mf.np64(sd[f"{p}.norm1.bias"])))  # fmt: skip
        self.n2 = (mf.put((dim,), dtype, lambda: mf.np64(sd[f"{p}.norm2.weight"])),
                   mf.put((dim,), dtype, lambda: mf.np64(sd[f"{p}.norm2.bias"])))  # fmt: skip
        self.m1 = mf.put((2 * dim, 2 * dim), dtype, lambda: mf.np64(sd[f"{p}.mlp.0.weight"]))
        self.m2 = mf.put((dim, 2 * dim), dtype, lambda: mf.np64(sd[f"{p}.mlp.2.weight"]))
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        """`[2, L, C]` (image 1, image 2): each image's tokens attend to the other's."""
        d = self.dim
        qkv = mx.addmm(self.qkv_b, x, self.qkv.T)
        q, kv = qkv[..., :d], qkv[::-1, :, d:]
        msg = attention(q[:, None], kv[:, None, :, :d], kv[:, None, :, d:], scale=self.scale)[:, 0]
        msg = mx.fast.layer_norm(mx.addmm(self.merge[1], msg, self.merge[0].T), *self.n1, 1e-5)
        y = mf.gelu(mx.concatenate([x, msg], axis=-1) @ self.m1.T) @ self.m2.T
        return x + mx.fast.layer_norm(y, *self.n2, 1e-5)


class _Refine:
    """`Refine`: [correlation | context | iteration context | flow | radius] -> 7 conv-LeakyReLU, a conv."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        def pick(rows: list[int]) -> Callable[[np.ndarray], np.ndarray]:
            return lambda w: w[:, rows]

        self.dtype = dtype
        self.c1 = mf.conv(sd, f"{p}.conv1.conv", dtype, inputs=pick(_STEP_IN), cin=len(_STEP_IN))
        self.c1_const = mf.conv(sd, f"{p}.conv1.conv", dtype, inputs=pick(_CONST_IN), cin=len(_CONST_IN))
        self.layers: list[Conv] = [mf.conv(sd, f"{p}.conv2.conv", dtype)]
        i = 0
        while f"{p}.conv_layers.{i}.conv.weight" in sd:
            self.layers.append(mf.conv(sd, f"{p}.conv_layers.{i}.conv", dtype))
            i += 1
        self.c3 = mf.conv(sd, f"{p}.conv3", dtype)
        self.step = stepped(self._step)

    def const(self, context: mx.array) -> mx.array:
        """The first conv's context and radius term."""
        radius = mx.full((*context.shape[:-1], 1), RADIUS, dtype=context.dtype)
        return self.c1_const(mx.concatenate([context, radius], axis=-1))

    def _step(
        self, it: mx.array, flow: mx.array, const: mx.array, f1: mx.array, f2: mx.array
    ) -> tuple[mx.array, mx.array]:
        _, h, w, _ = flow.shape
        dt = self.dtype
        coords = (mf.grid(h, w) + flow).transpose(0, 3, 1, 2)
        corr = ops.corr_windows(f1, [f2], coords, RADIUS, 1 / math.sqrt(f1.shape[-1])).transpose(0, 2, 3, 1)
        x = _leaky(self.c1(mx.concatenate([corr.astype(dt), it, flow.astype(dt)], axis=-1)) + const)
        for c in self.layers:
            x = _leaky(c(x))
        x = self.c3(x)
        return mx.clip(x[..., 2:], -4.0, 4.0), flow + x[..., :2].astype(mx.float32)

    def __call__(self, features: mx.array, context: mx.array, flow: mx.array, iters: int) -> mx.array:
        """`iters` refinements of `flow` `[1, h, w, 2]` (float32) with the pair's `features` `[2, h, w, 128]`."""
        const = self.const(context)
        it = mx.zeros((*context.shape[:-1], ITER_CONTEXT), dtype=self.dtype)
        f1, f2 = features[:1], features[1:]
        for _ in range(iters):
            it, flow = self.step(it, flow, const, f1, f2)
        return flow


class MlxNeuFlow:
    """NeuFlow v2 from the state dict `sd` in `dtype` (the name), `iters` refinements at 1/8; one pair per call."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        f32 = mx.float32
        self.iters = iters
        b = "backbone"
        self.b8_1 = _block(sd, f"{b}.block_8_1", f32, stride=4, padding=2)
        self.b8_2 = _block(sd, f"{b}.block_8_2", f32, stride=2, padding=2)
        self.cat8 = _block(sd, f"{b}.block_cat_8", f32, stride=1, padding=1)
        self.b16_1 = _block(sd, f"{b}.block_16_1", f32, stride=2, padding=2)
        self.b8_16 = _block(sd, f"{b}.block_8_16", f32, stride=2, padding=2)
        self.cat16 = _block(sd, f"{b}.block_cat_16", f32, stride=1, padding=1)
        self.layers = [_Layer(sd, f"cross_attn_s16.layers.{i}", dt) for i in range(2)]
        n = "cross_attn_s16.norm"

        def bn() -> tuple[np.ndarray, np.ndarray]:
            s = mf.np64(sd[f"{n}.weight"]) / np.sqrt(mf.np64(sd[f"{n}.running_var"]) + 1e-5)
            return s, mf.np64(sd[f"{n}.bias"]) - mf.np64(sd[f"{n}.running_mean"]) * s

        dim = FEATURE + CONTEXT
        self.bn = (mf.put((dim,), dt, lambda: bn()[0]), mf.put((dim,), dt, lambda: bn()[1]))
        self.merge8 = _merge(sd, "merge_s8", f32)
        self.cmerge8 = _merge(sd, "context_merge_s8", f32)
        self.refine16 = _Refine(sd, "refine_s16", dt)
        self.refine8 = _Refine(sd, "refine_s8", dt)
        self.conv8 = _block(sd, "conv_s8", dt, stride=8, padding=0)
        u = "upsample_s8"
        self.u1 = mf.conv(sd, f"{u}.conv1", dt)
        self.u2 = mf.conv(sd, f"{u}.conv2", dt)
        self.u3 = mf.conv(sd, f"{u}.conv3", dt, rows=mf.convex_rows(FACTOR))
        self._backbone = stepped(self._backbone_fn)
        self._s16 = stepped(self._s16_fn)
        self._s8 = stepped(self._s8_fn)
        self._head = stepped(self._head_fn)

    def _backbone_fn(self, img: mx.array) -> tuple[mx.array, mx.array]:
        """`[2, H, W, 3]` float32 -> 1/16 features (the last two channels: centred row, column) and 1/8 features."""
        img = mf.pyramid(img, 2)[1]
        x8 = self.b8_1(img)
        img = mf.pyramid(img, 2)[1]
        x8 = self.cat8(mx.concatenate([x8, self.b8_2(img)], axis=-1))
        img = mf.pyramid(img, 2)[1]
        x16 = self.cat16(mx.concatenate([self.b16_1(img), self.b8_16(x8)], axis=-1))
        n, h, w, _ = x16.shape
        ys = mx.broadcast_to((mx.arange(h, dtype=mx.float32) - h / 2)[:, None], (h, w))
        xs = mx.broadcast_to((mx.arange(w, dtype=mx.float32) - w / 2)[None], (h, w))
        pos = mx.broadcast_to(mx.stack([ys, xs], axis=-1)[None], (n, h, w, 2))
        return mx.concatenate([x16, pos], axis=-1), x8

    def _s16_fn(self, x16: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Cross-attention, the split and the global matching: 1/16 features, context and flow (float32)."""
        n, h, w, c = x16.shape
        x = x16.astype(self.dtype).reshape(n, h * w, c)
        for layer in self.layers:
            x = layer(x)
        x = (x * self.bn[0] + self.bn[1]).reshape(n, h, w, c)
        context, features = mf.relu(x[:1, ..., :CONTEXT]), x[..., CONTEXT:]
        f = features.astype(mx.float32).reshape(n, 1, h * w, FEATURE)
        g = mf.grid(h, w).reshape(1, 1, h * w, 2)
        v = mx.pad(g, [(0, 0), (0, 0), (0, 0), (0, FEATURE - 2)])  # a fused head size: no score matrix
        match = attention(f[:1], f[1:], v, scale=FEATURE**-0.5)[..., :2]
        return features, context, (match - g).reshape(1, h, w, 2)

    def _s8_fn(
        self, x8: mx.array, features16: mx.array, context16: mx.array, flow: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """The 1/8 features and context merged with the upsampled 1/16 ones, the flow upsampled."""
        f32, dt = mx.float32, self.dtype
        features = self.merge8(mx.concatenate([x8[..., CONTEXT:], _up2(features16).astype(f32)], axis=-1))
        context = mf.relu(x8[:1, ..., :CONTEXT])
        context = self.cmerge8(mx.concatenate([context, _up2(context16).astype(f32)], axis=-1))
        return features.astype(dt), context.astype(dt), _up2(flow) * 2

    def _head_fn(self, image1: mx.array, flow: mx.array) -> mx.array:
        feature = self.conv8(image1)
        mask = self.u3(mf.relu(self.u2(mf.relu(self.u1(mx.concatenate([flow.astype(self.dtype), feature], axis=-1))))))
        return mf.convex_upsample(flow, mask, FACTOR)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        x1 = image1.transpose(0, 2, 3, 1)
        x2 = image2.transpose(0, 2, 3, 1)
        x16, x8 = self._backbone(mx.concatenate([x1, x2], axis=0))
        features, context, flow = self._s16(x16)
        flow = self.refine16(features, context, flow, ITERS_S16)
        features, context, flow = self._s8(x8, features, context, flow)
        flow = self.refine8(features, context, flow, self.iters)
        up = self._head(x1.astype(self.dtype), flow)
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [0, 1]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
