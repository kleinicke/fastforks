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
from fastforks.kernels.graph_mlx.ops import bilinear_nhwc

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["mask_sigmoid_rows", "resize", "resize_argmax", "slide_scores", "steps_argmax", "upsample_argmax"]

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
        # weights in the map's own float dtype: float32 taps would promote a float16 map (and every map built on
        # it) to float32, twice the memory of the graph translation's resize
        x = a + (mx.take(x, i1, axis=axis) - a) * t.reshape(shape).astype(x.dtype)
    return x


def resize(x: mx.array, size: Sequence[int]) -> mx.array:
    """`[B, h, w, C]` to `[B, *size, C]`: bilinear, half-pixel centres, the values of torch's resize (lazy).

    One pass of the graph translation's NHWC kernel (`ops.bilinear_nhwc`: the four taps read straight from `x`,
    float32 accumulation, one rounding to `x`'s dtype); `_resize`'s axis-by-axis gathers materialise two maps per
    axis and round the float16 map twice.
    """
    h, w = x.shape[1:3]
    oh, ow = int(size[0]), int(size[1])
    if (h, w) == (oh, ow):
        return x
    if x.ndim != 4 or x.dtype not in (mx.float16, mx.bfloat16, mx.float32):  # noqa: PLR2004 - NHWC
        return _resize(x, size)
    return bilinear_nhwc(x, (oh, ow), (h / oh, w / ow), align=False)


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


_MASK_ROWS = metal_kernel(
    name="seg_mask_resize2_sigmoid",
    input_names=["x", "my0", "my1", "mwy", "mx0", "mx1", "mwx", "y0", "y1", "wy", "x0", "x1", "wx"],
    output_names=["out"],
    source="""
        // rows of the mask logits after both bilinear steps (as `seg_resize2_argmax`), then the sigmoid: one thread
        // per output pixel walks the queries, so neither resized grid exists
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y;
        if (ox >= OW || oy >= uint(y0_shape[0])) return;
        int W = x_shape[1];
        int ty[2] = {int(y0[oy]), int(y1[oy])}, tx[2] = {int(x0[ox]), int(x1[ox])};
        const device float* q[2][2][2][2];  // [mid row][mid col][source row][source col]
        float cy[2][2], cx[2][2];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int a = ty[i], b = tx[j];
                cy[i][j] = mwy[a];
                cx[i][j] = mwx[b];
                int r0 = my0[a], r1 = my1[a], c0 = mx0[b], c1 = mx1[b];
                q[i][j][0][0] = x + (size_t(r0) * W + c0) * C;
                q[i][j][0][1] = x + (size_t(r0) * W + c1) * C;
                q[i][j][1][0] = x + (size_t(r1) * W + c0) * C;
                q[i][j][1][1] = x + (size_t(r1) * W + c1) * C;
            }
        }
        float fx = wx[ox], fy = wy[oy];
        device T* o = out + (size_t(oy) * OW + ox) * C;
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
            o[c] = T(1.0f / (1.0f + exp(-(top + (bot - top) * fy))));
        }
    """,
)


def mask_sigmoid_rows(x: mx.array, sizes: Sequence[Sequence[int]], rows: tuple[int, int], dtype: mx.Dtype) -> mx.array:
    """Rows `[rows[0], rows[1])` of `sigmoid(resize(resize(x, mid)[:crop[0], :crop[1]], size))`: `[n * W, C]`, lazy.

    `sizes` is `(mid, crop, size)`. `x` `[h, w, C]` float32 mask logits (channels last); both steps bilinear with
    half-pixel centres, computed in float32 and rounded once to `dtype`. The Mask2Former-family semantic inference at
    the image size: the padded-size and the image-size logits of every query never exist (250 queries at
    1024 x 2048: 2 GB in float32).
    """
    h, w, c = x.shape
    mid, crop, size = sizes
    my0, my1, mwy = _taps(h, mid[0])
    mx0, mx1, mwx = _taps(w, mid[1])
    y0, y1, wy = (t[rows[0] : rows[1]] for t in _taps(crop[0], size[0]))
    x0, x1, wx = _taps(crop[1], size[1])
    n = rows[1] - rows[0]
    (out,) = _MASK_ROWS(
        inputs=[x, my0, my1, mwy, mx0, mx1, mwx, y0, y1, wy, x0, x1, wx],
        template=[("T", dtype), ("C", c), ("OW", size[1])],
        grid=(size[1], n, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(n * size[1], c)],
        output_dtypes=[dtype],
    )
    return out


def upsample_argmax(scores: torch.Tensor, sizes: Sequence[Sequence[int]]) -> torch.Tensor:
    """Per-pixel argmax `[B, H, W]` (int64) of `scores` resized through `sizes`; the first class wins ties.

    Every step but the last runs as array operations on the (small) intermediate scores; the last resize and the
    argmax are one kernel, so the full-resolution scores never exist.
    """
    return steps_argmax(to_mx(scores.float().permute(0, 2, 3, 1)), sizes)


def steps_argmax(x: mx.array, sizes: Sequence[Sequence[int]]) -> torch.Tensor:
    """`upsample_argmax` of float32 MLX scores `[B, h, w, C]` (a hand-written network's logits, no CPU round trip)."""
    if len(sizes) == 2 and not os.environ.get("FF_MLX_SEG_UNFUSED"):  # noqa: PLR2004 - the usual model + image resize
        return _fused2(x, sizes)
    for size in sizes[:-1]:
        x = _resize(x, size)
    return resize_argmax(x, sizes[-1])


def resize_argmax(x: mx.array, size: Sequence[int]) -> torch.Tensor:
    """Per-pixel argmax `[B, H, W]` (int64) of the float32 scores `[B, h, w, C]` resized to `size`, in one kernel."""
    b, h, w, c = x.shape
    oh, ow = size
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


_SLIDE = metal_kernel(
    name="seg_slide_resize",
    input_names=["x", "oy", "ox", "y0", "y1", "wy", "x0", "x1", "wx"],
    output_names=["out"],
    source="""
        // one thread per pixel of the sliding-window scores: each window that covers it resized there, then the mean
        uint px = thread_position_in_grid.x, py = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (px >= OW || py >= OH) return;
        int H = x_shape[1], W = x_shape[2];
        device float* o = out + ((size_t(n) * OH + py) * OW + px) * C;
        for (int c = 0; c < C; ++c) o[c] = 0.0f;
        float count = 0.0f;
        for (int k = 0; k < K; ++k) {
            int cy = int(py) - oy[k], cx = int(px) - ox[k];
            if (cy < 0 || cy >= CH || cx < 0 || cx >= CW) continue;
            const device float* base = x + (size_t(k) * B + n) * H * W * C;
            const device float* p00 = base + (size_t(y0[cy]) * W + x0[cx]) * C;
            const device float* p01 = base + (size_t(y0[cy]) * W + x1[cx]) * C;
            const device float* p10 = base + (size_t(y1[cy]) * W + x0[cx]) * C;
            const device float* p11 = base + (size_t(y1[cy]) * W + x1[cx]) * C;
            float fx = wx[cx], fy = wy[cy];
            for (int c = 0; c < C; ++c) {
                float top = p00[c] + (p01[c] - p00[c]) * fx;
                float bot = p10[c] + (p11[c] - p10[c]) * fx;
                o[c] += top + (bot - top) * fy;
            }
            count += 1.0f;
        }
        for (int c = 0; c < C; ++c) o[c] /= count;
    """,
)


def slide_scores(
    logits: Callable[[torch.Tensor], mx.array], x: torch.Tensor, crop: Sequence[int], stride: Sequence[int]
) -> mx.array | None:
    """The mmseg `slide_inference` (`seginfer.slide`) of a network with low-resolution logits, on MLX.

    `logits` maps crops `[n, 3, ch, cw]` to their logits `[n, gh, gw, C]` on MLX, which the network would resize to
    the crop (bilinear, `align_corners=False`). That resize and the windows' average are one kernel in float32, so
    neither the windows' crop-size logits nor their sum exist: `[B, h, w, C]` float32 scores at `x`'s size (lazy).
    Windows are batched per row, as `slide`. None if `x` is smaller than the crop (clipped windows: use `slide`).
    """
    (ch, cw), (sh, sw) = crop, stride
    b, _, h, w = x.shape
    if h < ch or w < cw:
        return None
    ys = [min(i * sh, h - ch) for i in range((h - ch + sh - 1) // sh + 1)]
    xs = [min(j * sw, w - cw) for j in range((w - cw + sw - 1) // sw + 1)]
    low = [logits(torch.cat([x[:, :, y : y + ch, x1 : x1 + cw] for x1 in xs])).astype(mx.float32) for y in ys]
    lows = mx.concatenate(low) if len(low) > 1 else low[0]
    _, gh, gw, c = lows.shape
    oy = mx.array(np.repeat(np.array(ys, np.int32), len(xs)))
    ox = mx.array(np.tile(np.array(xs, np.int32), len(ys)))
    (out,) = _SLIDE(
        inputs=[lows, oy, ox, *_taps(gh, ch), *_taps(gw, cw)],
        template=[("C", c), ("OH", h), ("OW", w), ("CH", ch), ("CW", cw), ("K", len(ys) * len(xs)), ("B", b)],
        grid=(w, h, b),
        threadgroup=(32, 8, 1),
        output_shapes=[(b, h, w, c)],
        output_dtypes=[mx.float32],
    )
    return out


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
