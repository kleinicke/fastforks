"""Twins' attention blocks with sine position terms (FlowFormer's cost encoder) in `mlx.core` operations.

Channels-last `[B, H, W, C]` maps; each block is timm's pre-norm `Block` (LayerNorm eps 1e-5, attention, residual,
GELU MLP, residual) around one of:

- `LocalBlock`: `LocallyGroupedAttnRPE`, attention within `ws` x `ws` windows of the zero-padded map; queries and keys
  see the window position's sine term, which with the query scale is folded into a per-position bias;
- `GlobalBlock`: `GlobalSubSampleAttnRPE`, every pixel's query against the keys of the `sr` x `sr` patch-conv
  subsampled map (LayerNorm after it), both with their grid's sine term (`_core.mlx.attention`, no score matrix).

The weights come from the state dict through `mlx_flow.put` (the conversion cache, `mlx_port.recorded`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import attention, mx
from fastforks.shared import mlx_flow as mf

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GlobalBlock", "LocalBlock", "grid_np", "norm", "sine", "sine_np"]

EPS = 1e-5


def sine(xy: mx.array, dim: int) -> mx.array:
    """Upstream `LinearPositionEmbeddingSine` of float32 `[..., 2]` (x, y) coordinates: `[..., dim]` float32."""
    f = mx.arange(dim // 4, dtype=mx.float32) * (3.14 / 200)  # upstream's 3.14, not pi
    x, y = xy[..., :1] * f, xy[..., 1:] * f
    return mx.concatenate([mx.sin(x), mx.cos(x), mx.sin(y), mx.cos(y)], axis=-1)


def sine_np(xy: np.ndarray, dim: int) -> np.ndarray:
    """`sine` in float64 numpy (the conversion's constants)."""
    f = np.arange(dim // 4) * (3.14 / 200)
    x, y = xy[..., :1] * f, xy[..., 1:] * f
    return np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=-1)


def grid_np(h: int, w: int) -> np.ndarray:
    """Upstream `coords_grid` as `[h * w, 2]` (x, y) float64, row major."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1).astype(np.float64)


def _grid(h: int, w: int) -> mx.array:
    return mf.grid(h, w).reshape(h * w, 2)


def norm(sd: Mapping[str, Any], name: str, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """The `LayerNorm` `name`'s weight and bias."""
    c = int(sd[f"{name}.weight"].shape[0])
    return (
        mf.put((c,), dtype, lambda: mf.np64(sd[f"{name}.weight"])),
        mf.put((c,), dtype, lambda: mf.np64(sd[f"{name}.bias"])),
    )


def _ln(x: mx.array, p: tuple[mx.array, mx.array]) -> mx.array:
    return mx.fast.layer_norm(x, p[0], p[1], EPS)


class _Block:
    """timm's pre-norm transformer block: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        self.dtype = dtype
        self.n1, self.n2 = norm(sd, f"{p}.norm1", dtype), norm(sd, f"{p}.norm2", dtype)
        self.c = int(sd[f"{p}.norm1.weight"].shape[0])
        self.heads = 8
        self.wp, self.bp = mf.linear(sd, f"{p}.attn.proj", dtype)
        self.w1, self.b1 = mf.linear(sd, f"{p}.mlp.fc1", dtype)
        self.w2, self.b2 = mf.linear(sd, f"{p}.mlp.fc2", dtype)

    def _attn(self, x: mx.array) -> mx.array:
        raise NotImplementedError

    def __call__(self, x: mx.array) -> mx.array:
        """`[B, H, W, C]` -> the same."""
        shape = x.shape
        x = x + (self._attn(_ln(x, self.n1)).reshape(-1, self.c) @ self.wp.T + self.bp).reshape(shape)
        h = mf.gelu(_ln(x, self.n2).reshape(-1, self.c) @ self.w1.T + self.b1)
        return x + (h @ self.w2.T + self.b2).reshape(shape)


def _qk_scaled(sd: Mapping[str, Any], p: str, c: int, heads: int) -> tuple[np.ndarray, np.ndarray]:
    """`attn.q` with the attention scale folded in and `attn.k`, stacked: `[2 C, C]` weight, `[2 C]` bias."""
    s = (c // heads) ** -0.5
    w = np.concatenate([mf.np64(sd[f"{p}.attn.q.weight"]) * s, mf.np64(sd[f"{p}.attn.k.weight"])])
    b = np.concatenate([mf.np64(sd[f"{p}.attn.q.bias"]) * s, mf.np64(sd[f"{p}.attn.k.bias"])])
    return w, b


class LocalBlock(_Block):
    """`Block` around `LocallyGroupedAttnRPE` (`ws` x `ws` windows, the padding zeros taking part as keys)."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype, ws: int = 7) -> None:
        """Convert the block `p` of `sd`."""
        super().__init__(sd, p, dtype)
        c, self.ws = self.c, ws
        self.wqk = mf.put((2 * c, c), dtype, lambda: _qk_scaled(sd, p, c, self.heads)[0])

        def bias() -> np.ndarray:  # the window position's sine term through q and k, plus their biases
            w, b = _qk_scaled(sd, p, c, self.heads)
            return sine_np(grid_np(ws, ws), c) @ w.T + b

        self.bqk = mf.put((ws * ws, 2 * c), dtype, bias)
        self.wv, self.bv = mf.linear(sd, f"{p}.attn.v", dtype)

    def _attn(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        ws, heads = self.ws, self.heads
        x = mx.pad(x, ((0, 0), (0, -h % ws), (0, -w % ws), (0, 0)))
        hp, wp = x.shape[1], x.shape[2]
        gh, gw = hp // ws, wp // ws
        t = x.reshape(b, gh, ws, gw, ws, c).transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        qk = t @ self.wqk.T + self.bqk
        v = mx.addmm(self.bv, t.reshape(-1, c), self.wv.T).reshape(t.shape)

        def heads_of(a: mx.array) -> mx.array:
            return a.reshape(-1, ws * ws, heads, c // heads).transpose(0, 2, 1, 3)

        o = attention(heads_of(qk[..., :c]), heads_of(qk[..., c:]), heads_of(v), scale=1.0)
        o = o.transpose(0, 2, 1, 3).reshape(b, gh, gw, ws, ws, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        return o[:, :h, :w]


class GlobalBlock(_Block):
    """`Block` around `GlobalSubSampleAttnRPE` (`sr` x `sr` subsampled keys and values)."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype, sr: int = 4) -> None:
        """Convert the block `p` of `sd`."""
        super().__init__(sd, p, dtype)
        c, self.sr = self.c, sr
        s = (c // self.heads) ** -0.5
        self.wq = mf.put((c, c), dtype, lambda: mf.np64(sd[f"{p}.attn.q.weight"]) * s)
        self.bq = mf.put((c,), dtype, lambda: mf.np64(sd[f"{p}.attn.q.bias"]) * s)
        self.wk, self.bk = mf.linear(sd, f"{p}.attn.k", dtype)
        self.wv, self.bv = mf.linear(sd, f"{p}.attn.v", dtype)
        self.sr_conv = mf.conv(sd, f"{p}.attn.sr", dtype, stride=sr, padding=0)
        self.sr_norm = norm(sd, f"{p}.attn.norm", dtype)

    def _attn(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        sr, heads, dt = self.sr, self.heads, self.dtype
        x = mx.pad(x, ((0, 0), (0, -h % sr), (0, -w % sr), (0, 0)))
        hp, wp = x.shape[1], x.shape[2]
        q = x.reshape(b, hp * wp, c) @ self.wq.T + (sine(_grid(hp, wp), c).astype(dt) @ self.wq.T + self.bq)
        xs = _ln(self.sr_conv(x), self.sr_norm).reshape(b, -1, c)
        pos = sine(_grid(hp // sr, wp // sr) * sr, c).astype(dt)
        k = (xs + pos) @ self.wk.T + self.bk
        v = xs @ self.wv.T + self.bv

        def heads_of(a: mx.array) -> mx.array:
            return a.reshape(b, -1, heads, c // heads).transpose(0, 2, 1, 3)

        o = attention(heads_of(q), heads_of(k), heads_of(v), scale=1.0)
        return o.transpose(0, 2, 1, 3).reshape(b, hp, wp, c)[:, :h, :w]
