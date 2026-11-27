"""Swin (v1) layers on Apple MLX, hand-written: the tokens `[h * w, C]` of one image, the run dtype.

The window shuffles of a Swin layer as gathers (`shared.maskformer.mlx_net`'s, shared with BiRefNet's port):

- LayerNorm + pad + cyclic shift + window partition is one kernel that gathers each window row through an index
  table (padding reads zeros: Swin v1 and transformers pad after the LayerNorm); window reverse + shift back + crop +
  residual + the second LayerNorm is the inverse gather in one kernel. Windows are ordered interior first, so only
  the border windows add the shift mask (-100, Swin's regions of the padded map).
- Window attention: the shared `window_attention` kernel (fused qkv read in place) where it fits
  (`graph_mlx.ops.window_attention_ok`: float16 12 x 12 windows of 32-wide heads), else
  `mx.fast.scaled_dot_product_attention` with the bias (and the mask) summed (float32: K and V of a window do not fit
  the kernel's threadgroup memory; 7 x 7 windows: 49 rows).

`Block` holds one layer's arrays, converted by its caller (`[in, out]` weights, the relative position bias gathered
to `[1, heads, T, T]`).
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import numpy as np

from fastforks._core.mlx import attention, metal_kernel, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_vit

__all__ = ["EPS", "Block", "Win", "gather_add_ln", "linear", "ln", "ln_gather", "windows"]

EPS = 1e-5  # Swin's LayerNorms (torch's default, transformers' layer_norm_eps)

_ROW = """
        uint lane = thread_index_in_simdgroup;
        size_t row = size_t(threadgroup_position_in_grid.x) * SG + simdgroup_index_in_threadgroup;
        if (row >= size_t(ROWS)) return;
"""

_LN = """
        float s = 0.0f;
        for (int i = 0; i < NPL; ++i) s += v[i];
        float mean = simd_sum(s) / C;
        float q = 0.0f;
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            float d = c < uint(C) ? v[i] - mean : 0.0f;
            q += d * d;
        }
        float rstd = rsqrt(simd_sum(q) / C + 1e-5f);  // EPS
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            if (c < uint(C)) o[c] = T((v[i] - mean) * rstd * float(w[c]) + float(b[c]));
        }
"""

_LN_GATHER = metal_kernel(
    name="maskformer_ln_gather",
    input_names=["x", "idx", "w", "b"],
    output_names=["out"],
    source=_ROW
    + """
        // window row `row` = LayerNorm(x[idx[row]]), zeros where idx < 0 (padding); one simdgroup per row, float32
        device T* o = out + row * C;
        int src = idx[row];
        float v[NPL];
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            v[i] = (src >= 0 && c < uint(C)) ? float(x[size_t(src) * C + c]) : 0.0f;
        }
        if (src < 0) {
            for (int i = 0; i < NPL; ++i) {
                uint c = lane + 32 * i;
                if (c < uint(C)) o[c] = T(0);
            }
            return;
        }
    """
    + _LN,
)

_GATHER_ADD_LN = metal_kernel(
    name="maskformer_gather_add_ln",
    input_names=["x", "a", "inv", "w", "b"],
    output_names=["hs", "out"],
    source=_ROW
    + """
        // token `row`: hs = x + a[inv[row]] (rounded to T, as the residual is stored), out = LayerNorm(hs)
        const device T* xr = x + row * C;
        const device T* ar = a + size_t(inv[row]) * C;
        device T* h = hs + row * C;
        device T* o = out + row * C;
        float v[NPL];
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            if (c < uint(C)) {
                T r = T(float(xr[c]) + float(ar[c]));
                h[c] = r;
                v[i] = float(r);
            } else {
                v[i] = 0.0f;
            }
        }
    """
    + _LN,
)


def _row_launch(rows: int, c: int, dtype: mx.Dtype) -> dict[str, Any]:
    sg = 8
    return {
        "template": [("T", dtype), ("C", c), ("NPL", -(-c // 32)), ("SG", sg), ("ROWS", rows)],
        "grid": (-(-rows // sg) * 32 * sg, 1, 1),
        "threadgroup": (32 * sg, 1, 1),
    }


def ln_gather(x: mx.array, idx: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """`LayerNorm(x[idx])` of tokens `[N, C]` (rows where `idx < 0` zero)."""
    rows, c = idx.shape[0], x.shape[1]
    return _LN_GATHER(
        inputs=[x, idx, w, b], output_shapes=[(rows, c)], output_dtypes=[x.dtype], **_row_launch(rows, c, x.dtype)
    )[0]


def gather_add_ln(x: mx.array, a: mx.array, inv: mx.array, w: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    """`hs = x + a[inv]` and `LayerNorm(hs)` of tokens `[N, C]`."""
    n, c = x.shape
    hs, y = _GATHER_ADD_LN(
        inputs=[x, a, inv, w, b], output_shapes=[(n, c), (n, c)], output_dtypes=[x.dtype, x.dtype],
        **_row_launch(n, c, x.dtype),
    )  # fmt: skip
    return hs, y


def ln(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """LayerNorm over the last axis (`EPS`): the row kernel up to `ROW_LN_MAX` channels, else MLX's."""
    if x.shape[-1] <= ops.ROW_LN_MAX:
        return ops.row_layer_norm(x, w, b, EPS)
    return mx.fast.layer_norm(x, w, b, EPS)


def linear(x: mx.array, wb: tuple[mx.array, mx.array | None]) -> mx.array:
    """`x @ w (+ b)` of an `[in, out]` weight."""
    w, b = wb
    return x @ w if b is None else mx.addmm(b, x, w)


class Win(NamedTuple):
    """Window layout of one Swin layer type on one feature-map size."""

    windows: int
    interior: int  # windows that need no shift mask (they come first)
    idx: mx.array  # [rows] int32 source token per window row, -1 for padding
    inv: mx.array  # [tokens] int32 window row per token
    mask: mx.array | None  # [border windows, 1, T, T] shift mask (-100 / 0) in the run dtype


@functools.lru_cache(maxsize=32)
def windows(h: int, w: int, ws: int, s: int, dtype: mx.Dtype) -> Win:
    """The layout of `ws x ws` windows shifted by `s` on an `h x w` map (padded to window multiples)."""
    hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
    nh, nw = hp // ws, wp // ws
    if s > 0:  # interior first, then the last column, then the last row (each row-major)
        order = [y * nw + x for y in range(nh - 1) for x in range(nw - 1)]
        order += [y * nw + nw - 1 for y in range(nh - 1)] + [(nh - 1) * nw + x for x in range(nw)]
        interior = (nh - 1) * (nw - 1)
    else:
        order, interior = list(range(nh * nw)), nh * nw
    order_a = np.asarray(order)
    wy, wx = order_a // nw, order_a % nw
    iy, ix = np.repeat(np.arange(ws), ws), np.tile(np.arange(ws), ws)
    y, x = wy[:, None] * ws + iy[None], wx[:, None] * ws + ix[None]  # shifted padded-grid coordinates [windows, T]
    sy, sx = (y + s) % hp, (x + s) % wp  # torch.roll(-s): out[i] = in[i + s]
    valid = ((sy < h) & (sx < w)).reshape(-1)
    src = (sy * w + sx).reshape(-1)
    inv = np.zeros(h * w, dtype=np.int32)
    inv[src[valid]] = np.nonzero(valid)[0]
    mask = None
    if s > 0 and interior < nh * nw:  # Swin's (and transformers' `get_attn_mask`) regions, border windows only
        reg = ((y >= hp - ws).astype(int) + (y >= hp - s)) * 3 + (x >= wp - ws) + (x >= wp - s)
        rb = reg[interior:]
        m = np.where(rb[:, None, :] != rb[:, :, None], -100.0, 0.0)[:, None]
        mask = mx.array(m.astype(np.float32)).astype(dtype)
    return Win(nh * nw, interior, mx.array(np.where(valid, src, -1).astype(np.int32)), mx.array(inv), mask)


def _fused(t: int, d: int, dtype: mx.Dtype) -> bool:
    """Whether `ops.window_attention` takes `t`-token windows of `d`-wide heads (`window_attention_ok`'s rule)."""
    return t % 8 == 0 and d % 8 == 0 and t <= 256 and d <= 128 and 2 * t * d * dtype.size <= ops.WIN_ATTN_TG_BYTES  # noqa: PLR2004


class Block:
    """One Swin layer: `x + attn(LN1(x))`, then `+ mlp(LN2(.))`, on tokens `[h * w, C]`.

    Attributes (set by the caller, arrays in the run dtype): `window`, `shift`, `heads`, `head_dim`, `scale`;
    `qkv`, `o`, `fc1`, `fc2`: `([in, out], bias)`; `rpb`: `[1, heads, T, T]`; `ln1`, `ln2`: `(weight, bias)`.
    """

    window: int
    shift: int
    heads: int
    head_dim: int
    scale: float
    qkv: tuple[mx.array, mx.array | None]
    o: tuple[mx.array, mx.array | None]
    fc1: tuple[mx.array, mx.array | None]
    fc2: tuple[mx.array, mx.array | None]
    rpb: mx.array
    ln1: tuple[mx.array, mx.array]
    ln2: tuple[mx.array, mx.array]

    def _attention(self, qkv: mx.array, g: Win) -> mx.array:
        """Fused qkv `[windows, T, 3, heads, D]` -> `[windows * T, C]` (interior windows: no shift mask)."""
        c = self.heads * self.head_dim
        parts = [(qkv, [self.rpb])] if g.mask is None else [(qkv[g.interior :], [self.rpb, g.mask])]
        if g.mask is not None and g.interior:
            parts.insert(0, (qkv[: g.interior], [self.rpb]))
        fused = _fused(qkv.shape[1], self.head_dim, qkv.dtype)
        out = []
        for part, masks in parts:
            if fused:
                o = ops.window_attention(part, None, None, self.scale, masks)
            else:
                q, k, v = (part[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
                bias = masks[0] if len(masks) == 1 else masks[0] + masks[1]
                o = attention(q, k, v, scale=self.scale, mask=bias)
            out.append(o.transpose(0, 2, 1, 3).reshape(-1, c))
        return out[0] if len(out) == 1 else mx.concatenate(out)

    def __call__(self, x: mx.array, hw: tuple[int, int]) -> mx.array:
        """Tokens `[h * w, C]` -> the layer's output."""
        g = windows(*hw, self.window, self.shift, x.dtype)
        t = self.window * self.window
        win = ln_gather(x, g.idx, *self.ln1)
        qkv = linear(win, self.qkv).reshape(g.windows, t, 3, self.heads, self.head_dim)
        o = linear(self._attention(qkv, g), self.o)
        hs, y = gather_add_ln(x, o, g.inv, *self.ln2)
        return linear(mlx_vit.gelu(linear(y, self.fc1)), self.fc2) + hs
