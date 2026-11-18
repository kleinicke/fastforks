"""Segmentation post-processing on MLX: the logits' bilinear resize steps and the per-pixel argmax run on the GPU.

An MLX runner returns its network's low-resolution logits as CPU tensors; resizing 150 classes to 1024 x 2048 in
PyTorch on the CPU would cost more than the network. `fastforks.shared.seginfer.upsample` / `upsample_argmax` call this
while an `mlx_wrap.port` runner is running (`mlx_wrap.POST_ON_MLX`): the same half-pixel-centre bilinear steps as
`F.interpolate(mode="bilinear", align_corners=False)` (no antialiasing), in float32, class chunk by chunk.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import metal_kernel, mx, to_mx

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["upsample_argmax"]

_CHUNK_ELEMENTS = 1 << 26  # float32 elements of one class chunk's resized scores (256 MB)


def _taps(n_in: int, n_out: int) -> tuple[mx.array, mx.array, mx.array]:
    """Half-pixel-centre bilinear taps of one axis `(i0, i1, weight of i1)`, as torch computes them (float32)."""
    src = np.maximum((np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * np.float32(n_in / n_out) - 0.5, 0)
    i0 = np.minimum(src.astype(np.int32), n_in - 1)
    return (
        mx.array(i0),
        mx.array(np.minimum(i0 + 1, n_in - 1).astype(np.int32)),
        mx.array((src - i0).astype(np.float32)),
    )


def _resize(x: mx.array, size: Sequence[int]) -> mx.array:
    """`[B, h, w, C]` to `[B, *size, C]`, one axis at a time (the same values as torch's bilinear resize)."""
    for axis, n in ((1, size[0]), (2, size[1])):
        if x.shape[axis] == n:
            continue
        i0, i1, t = _taps(x.shape[axis], n)
        shape = [1, 1, 1, 1]
        shape[axis] = n
        a = mx.take(x, i0, axis=axis)
        x = a + (mx.take(x, i1, axis=axis) - a) * t.reshape(shape)
    return x


def _steps(x: mx.array, sizes: Sequence[Sequence[int]]) -> mx.array:
    for size in sizes:
        x = _resize(x, size)
    return x


_ARGMAX = metal_kernel(
    name="seg_resize_argmax",
    input_names=["x", "y0", "y1", "wy", "x0", "x1", "wx"],
    output_names=["out"],
    source="""
        // one thread per output pixel: the last bilinear step's four taps of every class, running first-wins maximum
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (ox >= OW || oy >= OH) return;
        int H = x_shape[1], W = x_shape[2];
        const device float* base = x + size_t(n) * H * W * C;
        const device float* p00 = base + (size_t(y0[oy]) * W + x0[ox]) * C;
        const device float* p01 = base + (size_t(y0[oy]) * W + x1[ox]) * C;
        const device float* p10 = base + (size_t(y1[oy]) * W + x0[ox]) * C;
        const device float* p11 = base + (size_t(y1[oy]) * W + x1[ox]) * C;
        float fx = wx[ox], fy = wy[oy];
        float best = -INFINITY;
        int arg = 0;
        for (int c = 0; c < C; ++c) {
            float top = p00[c] + (p01[c] - p00[c]) * fx;
            float bot = p10[c] + (p11[c] - p10[c]) * fx;
            float v = top + (bot - top) * fy;
            if (v > best || (v != v && best == best)) { best = v; arg = c; }
        }
        out[(size_t(n) * OH + oy) * OW + ox] = arg;
    """,
)


_ARGMAX2 = metal_kernel(
    name="seg_resize2_argmax",
    input_names=["x", "my0", "my1", "mwy", "mx0", "mx1", "mwx", "y0", "y1", "wy", "x0", "x1", "wx"],
    output_names=["out"],
    source="""
        // both bilinear steps and the argmax in one pass: each of the last step's four taps is itself a step-one
        // resize (rows then columns, as `_resize`) of the low-resolution scores, so the intermediate never exists
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (ox >= OW || oy >= OH) return;
        int H = x_shape[1], W = x_shape[2];
        const device float* base = x + size_t(n) * H * W * C;
        int ty[2] = {int(y0[oy]), int(y1[oy])}, tx[2] = {int(x0[ox]), int(x1[ox])};
        const device float* q[2][2][2][2];  // [mid row][mid col][source row][source col]
        float cy[2][2], cx[2][2];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int a = ty[i], b = tx[j];
                cy[i][j] = mwy[a];
                cx[i][j] = mwx[b];
                int r0 = my0[a], r1 = my1[a], c0 = mx0[b], c1 = mx1[b];
                q[i][j][0][0] = base + (size_t(r0) * W + c0) * C;
                q[i][j][0][1] = base + (size_t(r0) * W + c1) * C;
                q[i][j][1][0] = base + (size_t(r1) * W + c0) * C;
                q[i][j][1][1] = base + (size_t(r1) * W + c1) * C;
            }
        }
        float fx = wx[ox], fy = wy[oy];
        float best = -INFINITY;
        int arg = 0;
        for (int c = 0; c < C; ++c) {
            float m[2][2];
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    float l = q[i][j][0][0][c] + (q[i][j][1][0][c] - q[i][j][0][0][c]) * cy[i][j];
                    float r = q[i][j][0][1][c] + (q[i][j][1][1][c] - q[i][j][0][1][c]) * cy[i][j];
                    m[i][j] = l + (r - l) * cx[i][j];
                }
            }
            float top = m[0][0] + (m[0][1] - m[0][0]) * fx;
            float bot = m[1][0] + (m[1][1] - m[1][0]) * fx;
            float v = top + (bot - top) * fy;
            if (v > best || (v != v && best == best)) { best = v; arg = c; }
        }
        out[(size_t(n) * OH + oy) * OW + ox] = arg;
    """,
)


def upsample_argmax(scores: torch.Tensor, sizes: Sequence[Sequence[int]]) -> torch.Tensor:
    """Per-pixel argmax `[B, H, W]` (int64) of `scores` resized through `sizes`; the first class wins ties.

    Every step but the last runs as array operations on the (small) intermediate scores; the last resize and the
    argmax are one kernel, so the full-resolution scores never exist.
    """
    x = to_mx(scores.float().permute(0, 2, 3, 1))
    if len(sizes) == 2 and not os.environ.get("FF_MLX_SEG_UNFUSED"):  # noqa: PLR2004 - the usual model + image resize
        return _fused2(x, sizes)
    for size in sizes[:-1]:
        x = _resize(x, size)
    b, h, w, c = x.shape
    oh, ow = sizes[-1]
    y0, y1, wy = _taps(h, oh)
    x0, x1, wx = _taps(w, ow)
    (out,) = _ARGMAX(
        inputs=[x, y0, y1, wy, x0, x1, wx],
        template=[("C", c), ("OH", oh), ("OW", ow)],
        grid=(ow, oh, b),
        threadgroup=(32, 8, 1),
        output_shapes=[(b, oh, ow)],
        output_dtypes=[mx.int32],
    )
    mx.eval(out)
    return torch.from_numpy(np.array(out)).to(torch.int64)


def _fused2(x: mx.array, sizes: Sequence[Sequence[int]]) -> torch.Tensor:
    b, h, w, c = x.shape
    (mh, mw), (oh, ow) = sizes
    my0, my1, mwy = _taps(h, mh)
    mx0, mx1, mwx = _taps(w, mw)
    y0, y1, wy = _taps(mh, oh)
    x0, x1, wx = _taps(mw, ow)
    (out,) = _ARGMAX2(
        inputs=[x, my0, my1, mwy, mx0, mx1, mwx, y0, y1, wy, x0, x1, wx],
        template=[("C", c), ("OH", oh), ("OW", ow)],
        grid=(ow, oh, b),
        threadgroup=(32, 8, 1),
        output_shapes=[(b, oh, ow)],
        output_dtypes=[mx.int32],
    )
    mx.eval(out)
    return torch.from_numpy(np.array(out)).to(torch.int64)
