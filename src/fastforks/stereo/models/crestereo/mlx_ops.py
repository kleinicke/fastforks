"""CREStereo on Apple MLX, portable mode: the operations around the convolutions in MLX array ops (no custom Metal).

`fastforks.kernels.crestereo_mlx.ops` has the same functions as Metal kernels (kernels mode).
"""
# ruff: noqa: ANN001, ANN201, ANN202, D103, D205, PLR0913, PLR0917 - array-in / array-out maths functions
# (positional arrays, as the torch forward they mirror); shapes are in the docstrings

from __future__ import annotations

import numpy as np

from fastforks._core.mlx import mx

GROUPS = 4  # correlation groups (upstream AGCL)
SEARCH = 9  # window positions: 3x3 or 1x9
HDIM = 128  # GRU hidden channels


def instance_norm_relu(x, add=None, eps=1e-5):
    """`relu(instance_norm(x) [+ add])` per image and channel. The statistics are float32 means of x and x^2 (one
    cast of the map), the normalisation itself one multiply-add in the map's dtype.
    """
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=(1, 2), keepdims=True)
    var = mx.maximum(mx.mean(xf * xf, axis=(1, 2), keepdims=True) - mean * mean, 0)
    scale = mx.rsqrt(var + eps)
    y = x * scale.astype(x.dtype) - (mean * scale).astype(x.dtype)
    return mx.maximum(y if add is None else y + add, 0)


def coords_grid(h, w):
    ys, xs = mx.meshgrid(mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32), indexing="ij")
    return mx.stack([xs, ys], axis=-1)[None]


def sample(fmap, coords):
    """Torch `grid_sample(align_corners=True, padding_mode="zeros")` at pixel coords `[N, h, w, 2]` (x, y) fp32:
    `[N, h, w, C]` float32 (the four corners are read in the map's dtype and blended in fp32).
    """
    n, h, w, c = fmap.shape
    flat = fmap.reshape(n * h * w, c)
    x, y = coords[..., 0], coords[..., 1]
    x0f, y0f = mx.floor(x), mx.floor(y)
    fx, fy = x - x0f, y - y0f
    x0, y0 = x0f.astype(mx.int32), y0f.astype(mx.int32)
    out = None
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)
        v = flat[mx.clip(yi, 0, h - 1) * w + mx.clip(xi, 0, w - 1)].astype(mx.float32) * (wt * ok)[..., None]
        out = v if out is None else out + v
    return out


def _window(small):
    return (1, 1) if small else (0, 4)


def corr_iter(f1, f2, flow, small):
    """Warp `f2` by the flow, group correlation with `f1` over a 3x3 / 1x9 window of the (replicate padded) warped
    map: `[1, H, W, 36]`, channel `group * 9 + window index`.

    float32: one einsum per window position (no product array is materialised). float16: the same dot products as
    batched `[1, 64] @ [64, 1]` matmuls on float16 operands, which halves the bytes moved (this op is memory bound).
    """
    n, h, w, c = f1.shape
    warped = sample(f2, coords_grid(h, w) + flow)
    ry, rx = _window(small)
    half = f1.dtype == mx.float16
    padded = mx.pad(warped.astype(f1.dtype), [(0, 0), (ry, ry), (rx, rx), (0, 0)], mode="edge")
    shifts = [padded[:, dy : dy + h, dx : dx + w] for dy in range(2 * ry + 1) for dx in range(2 * rx + 1)]
    if half:
        left = f1.reshape(n * h * w * GROUPS, 1, c // GROUPS)
        corrs = [(left @ s.reshape(n * h * w * GROUPS, c // GROUPS, 1)).reshape(n, h, w, GROUPS) for s in shifts]
    else:
        left = f1.reshape(n, h, w, GROUPS, c // GROUPS)
        corrs = [mx.einsum("nhwgc,nhwgc->nhwg", left, s.reshape(n, h, w, GROUPS, c // GROUPS)) for s in shifts]
    return (mx.stack(corrs, axis=-1) * (GROUPS / c)).reshape(n, h, w, GROUPS * SEARCH)


def corr_offset(f1, f2, offs, flow, small):
    """Group correlation at the window positions shifted by the learned offsets `[1, H, W, 18]` (window * 2 + xy)."""
    n, h, w, c = f1.shape
    ry, rx = _window(small)
    win = np.array([(dx, dy) for dy in range(-ry, ry + 1) for dx in range(-rx, rx + 1)], np.float32)
    coords = (coords_grid(h, w) + flow)[:, :, :, None] + (mx.array(win) + offs.reshape(n, h, w, SEARCH, 2))
    warped = sample(f2, coords.reshape(n, h, w * SEARCH, 2)).reshape(n, h, w, SEARCH, GROUPS, c // GROUPS)
    corr = (f1.astype(mx.float32).reshape(n, h, w, 1, GROUPS, c // GROUPS) * warped).mean(axis=-1)
    return corr.swapaxes(3, 4).reshape(n, h, w, GROUPS * SEARCH)


def gru(p, hidden, motion, flow, czr1, cq1, czr2, cq2):
    """Both `SepConvGRU` halves. `motion` `[.., 128]` has zeros where the flow belongs (channels 126-127)."""
    motion = mx.concatenate([motion[..., : HDIM - 2], flow.astype(motion.dtype)], axis=-1)
    for half, pad, czr, cq in ((1, (0, 2), czr1, cq1), (2, (2, 0), czr2, cq2)):
        zr = mx.sigmoid(mx.conv2d(mx.concatenate([hidden, motion], -1), p[f"gru{half}.zr.weight"], padding=pad) + czr)
        z, r = zr[..., :HDIM], zr[..., HDIM:]
        q = mx.tanh(mx.conv2d(mx.concatenate([r * hidden, motion], -1), p[f"gru{half}.q.weight"], padding=pad) + cq)
        hidden = (1 - z) * hidden + z * q
    return hidden


def convex_upsample(flow, mask, rate=4):
    """`[1, H, W, 2]` + mask logits `[1, H, W, 9 * rate^2]` -> `[1, rate H, rate W, 2]` (float32)."""
    n, h, w, _ = flow.shape
    m = mx.softmax(mask.reshape(n, h, w, 9, rate * rate), axis=3)
    fp = mx.pad(rate * flow, [(0, 0), (1, 1), (1, 1), (0, 0)])
    u = mx.stack([fp[:, dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)], axis=3)  # [N, H, W, 9, 2]
    up = mx.einsum("nhwka,nhwkc->nhwac", m, u).reshape(n, h, w, rate, rate, 2)
    return up.transpose(0, 1, 3, 2, 4, 5).reshape(n, h * rate, w * rate, 2)
