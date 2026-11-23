"""Channels-last depthwise convolution and global pooling on Apple MLX (`fastforks.shared.mlx_graph`, kernels mode).

MLX's grouped convolution runs a depthwise layer as many tiny convolutions; one thread per output element with the
window held in registers is several times faster for the 3x3 .. 7x7 kernels of mobile and ConvNeXt-style networks.
Accumulation is float32, the result is rounded once to the activation dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks._core.mlx import metal_kernel, mx

if TYPE_CHECKING:
    from collections.abc import Callable

_DW = metal_kernel(
    name="graph_dwconv",
    input_names=["x", "w", "b"],
    output_names=["out"],
    source="""
        // one thread: 4 adjacent channels of R neighbouring output pixels of one row; fp32 accumulation, one rounding
        uint c4 = thread_position_in_grid.x, tile = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (c4 >= C / 4) return;
        int H = x_shape[1], W = x_shape[2];
        int tiles_w = (OWT + R - 1) / R;
        int oh = tile / tiles_w, ow0 = (tile % tiles_w) * R;
        const device vec<T, 4>* x4 = (const device vec<T, 4>*)x;
        const device vec<T, 4>* w4 = (const device vec<T, 4>*)w;
        float4 acc[R];
        for (int r = 0; r < R; ++r) acc[r] = float4(((const device vec<T, 4>*)b)[c4]);
        for (int i = 0; i < KH; ++i) {
            int ih = oh * SH - PH + i * DH;
            if (ih < 0 || ih >= H) continue;
            size_t row = (size_t(n) * H + ih) * W;
            if (SW == 1 && DW == 1) {  // each input pixel is loaded once for all R outputs
                float4 wv[KW];
                for (int j = 0; j < KW; ++j) wv[j] = float4(w4[(i * KW + j) * (C / 4) + c4]);
                for (int t = 0; t < KW + R - 1; ++t) {
                    int iw = ow0 - PW + t;
                    if (iw < 0 || iw >= W) continue;
                    float4 xv = float4(x4[(row + iw) * (C / 4) + c4]);
                    for (int r = 0; r < R; ++r) {
                        int j = t - r;
                        if (j >= 0 && j < KW) acc[r] += xv * wv[j];
                    }
                }
            } else {
                for (int j = 0; j < KW; ++j) {
                    float4 wv = float4(w4[(i * KW + j) * (C / 4) + c4]);
                    for (int r = 0; r < R; ++r) {
                        int iw = (ow0 + r) * SW - PW + j * DW;
                        if (iw < 0 || iw >= W) continue;
                        acc[r] += float4(x4[(row + iw) * (C / 4) + c4]) * wv;
                    }
                }
            }
        }
        for (int r = 0; r < R; ++r)
            if (ow0 + r < OWT)
                ((device vec<T, 4>*)out)[((size_t(n) * OHT + oh) * OWT + ow0 + r) * (C / 4) + c4] = vec<T, 4>(acc[r]);
    """,
)


def depthwise(  # noqa: PLR0913
    x: mx.array, w: mx.array, b: mx.array, stride: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int]
) -> mx.array:
    """NHWC depthwise convolution (channels a multiple of 4). `w` is `[kh, kw, C]`, `b` is `[C]`.

    Mixed dtypes promote as in `mx.conv2d` (the kernel reads all three as one type: a float32 activation against float16
    weights read them as garbage, FoundationStereo's float16 7x7 depthwise convs after a float32 norm).
    """
    t = mx.result_type(x.dtype, w.dtype, b.dtype)
    x, w, b = x.astype(t), w.astype(t), b.astype(t)
    n, h, wd, c = x.shape
    kh, kw = w.shape[0], w.shape[1]
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    c4 = c // 4
    tg = min(c4, 32)
    r = 4 if oh * ow >= 16 else 1  # outputs per thread along a row
    return _DW(
        inputs=[x, w, b],
        template=[
            ("T", x.dtype), ("C", c), ("KH", kh), ("KW", kw), ("SH", stride[0]), ("SW", stride[1]),
            ("PH", padding[0]), ("PW", padding[1]), ("DH", dilation[0]), ("DW", dilation[1]),
            ("OHT", oh), ("OWT", ow), ("R", r),
        ],
        grid=(-(-c4 // tg) * tg, oh * -(-ow // r), n),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n, oh, ow, c)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip


_POOL = metal_kernel(
    name="graph_global_avg",
    input_names=["x"],
    output_names=["out"],
    source="""
        size_t channel = thread_position_in_grid.x, batch = thread_position_in_grid.y;
        size_t pixels = size_t(x_shape[1]) * x_shape[2], channels = x_shape[3];
        size_t base = batch * pixels * channels + channel;
        float total = 0;
        for (size_t i = 0; i < pixels; ++i) total += float(x[base + i * channels]);
        out[batch * channels + channel] = T(total / float(pixels));
    """,
)


def global_avg(x: mx.array) -> mx.array:
    """Average each channel of an NHWC map in fp32: `[N, H, W, C]` -> `[N, C]`."""
    n, _, _, c = x.shape
    return _POOL(
        inputs=[x], template=[("T", x.dtype)], grid=(c, n, 1), threadgroup=(min(256, c), 1, 1),
        output_shapes=[(n, c)], output_dtypes=[x.dtype],
    )[0]  # fmt: skip


_STATS = metal_kernel(
    name="graph_channel_stats",
    input_names=["x"],
    output_names=["out"],
    source="""
        // partial sums of one channel over every G-th row, shifted by the first row's value (no cancellation)
        uint c = thread_position_in_grid.x, g = thread_position_in_grid.y;
        if (c >= C) return;
        float piv = float(x[c]);
        float s = 0, q = 0;
        for (size_t p = g; p < P; p += G) {
            float v = float(x[p * C + c]) - piv;
            s += v;
            q += v * v;
        }
        out[(size_t(g) * C + c) * 2] = s;
        out[(size_t(g) * C + c) * 2 + 1] = q;
    """,
)


def channel_stats(x: mx.array) -> tuple[mx.array, mx.array]:
    """Per-channel mean and (biased) variance of a `[P, C]` array in float32, one pass over the data."""
    p, c = x.shape
    g = max(1, min(1024, p // 256))
    part = _STATS(
        inputs=[x], template=[("T", x.dtype), ("C", c), ("P", p), ("G", g)], grid=(-(-c // 32) * 32, g, 1),
        threadgroup=(32, 1, 1), output_shapes=[(g, c, 2)], output_dtypes=[mx.float32],
    )[0]  # fmt: skip
    tot = part.sum(axis=0) / p  # [C, 2]: mean and mean of squares of the shifted values
    piv = x[0].astype(mx.float32)
    return piv + tot[:, 0], mx.maximum(tot[:, 1] - mx.square(tot[:, 0]), 0)


_GRID = metal_kernel(
    name="graph_grid_sample",
    input_names=["x", "grid"],
    output_names=["out"],
    source="""
        // one thread: VEC channels of one output pixel; the four taps are read straight from the NHWC map, fp32 weights
        uint cv = thread_position_in_grid.x, pix = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (cv >= C / VEC) return;
        size_t gi = (size_t(n) * HO * WO + pix) * 2;
        float gx = float(grid[gi]), gy = float(grid[gi + 1]);
        float ix = ALIGN ? (gx + 1.0f) * float(W - 1) * 0.5f : ((gx + 1.0f) * float(W) - 1.0f) * 0.5f;
        float iy = ALIGN ? (gy + 1.0f) * float(H - 1) * 0.5f : ((gy + 1.0f) * float(H) - 1.0f) * 0.5f;
        if (BORDER) {
            ix = clamp(ix, 0.0f, float(W - 1));
            iy = clamp(iy, 0.0f, float(H - 1));
        }
        float fx = floor(ix), fy = floor(iy);
        float ax = ix - fx, ay = iy - fy;
        int x0 = int(fx), y0 = int(fy);
        float4 acc = float4(0.0f);
        const device vec<T, VEC>* x4 = (const device vec<T, VEC>*)x;
        for (int dy = 0; dy < 2; ++dy) {
            int yy = y0 + dy;
            if (yy < 0 || yy >= H) continue;
            float wy = dy ? ay : 1.0f - ay;
            for (int dx = 0; dx < 2; ++dx) {
                int xx = x0 + dx;
                if (xx < 0 || xx >= W) continue;
                float wgt = wy * (dx ? ax : 1.0f - ax);
                vec<T, VEC> v = x4[((size_t(n) * H + yy) * W + xx) * (C / VEC) + cv];
                for (int k = 0; k < VEC; ++k) acc[k] += float(v[k]) * wgt;
            }
        }
        vec<T, VEC> o;
        for (int k = 0; k < VEC; ++k) o[k] = T(acc[k]);
        ((device vec<T, VEC>*)out)[(size_t(n) * HO * WO + pix) * (C / VEC) + cv] = o;
    """,
)


def grid_sample(x: mx.array, grid: mx.array, *, align_corners: bool, border: bool) -> mx.array:
    """Bilinear `grid_sample` of an NHWC map by a `[N, HO, WO, 2]` grid; `[N, HO, WO, C]` (zeros or border padding)."""
    n, h, w, c = x.shape
    ho, wo = grid.shape[1], grid.shape[2]
    vec = 4 if c % 4 == 0 else 2 if c % 2 == 0 else 1
    cv = c // vec
    tg = min(cv, 32)
    return _GRID(
        inputs=[mx.contiguous(x), mx.contiguous(grid)],
        template=[("T", x.dtype), ("C", c), ("H", h), ("W", w), ("HO", ho), ("WO", wo), ("VEC", vec),
                  ("ALIGN", align_corners), ("BORDER", border)],
        grid=(-(-cv // tg) * tg, ho * wo, n), threadgroup=(tg, 1, 1),
        output_shapes=[(n, ho, wo, c)], output_dtypes=[x.dtype],
    )[0]  # fmt: skip


_BILINEAR = metal_kernel(
    name="graph_bilinear",
    input_names=["x", "r"],
    output_names=["out"],
    source="""
        // one thread per output pixel of one plane: the four taps, fp32 weights, one rounding (no intermediates)
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y, plane = thread_position_in_grid.z;
        if (ox >= OW || oy >= OH) return;
        float ys = ALIGN ? float(oy) * r[0] : max((float(oy) + 0.5f) * r[0] - 0.5f, 0.0f);
        float xs = ALIGN ? float(ox) * r[1] : max((float(ox) + 0.5f) * r[1] - 0.5f, 0.0f);
        int y0 = min(int(ys), H - 1), x0 = min(int(xs), W - 1);
        int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float wy = ys - float(y0), wx = xs - float(x0);
        const device T* p = x + size_t(plane) * H * W;
        float top = float(p[y0 * W + x0]) * (1.0f - wx) + float(p[y0 * W + x1]) * wx;
        float bot = float(p[y1 * W + x0]) * (1.0f - wx) + float(p[y1 * W + x1]) * wx;
        out[(size_t(plane) * OH + oy) * OW + ox] = T(top * (1.0f - wy) + bot * wy);
    """,
)


def bilinear(x: mx.array, size: tuple[int, int], ratios: tuple[float, float], *, align: bool) -> mx.array:
    """Bilinear resize of an NCHW map to `size` (torch's `upsample_bilinear2d`): `ratios` = input / output per axis.

    With `align` the source coordinate is `i * ratio` (`ratio` = (in - 1) / (out - 1)), else the half-pixel centre
    `(i + 0.5) * ratio - 0.5`. Accumulation is float32, rounded once; the four taps are read straight from `x`.
    """
    n, c, h, w = x.shape
    return _BILINEAR(
        inputs=[x, mx.array(ratios, dtype=mx.float32)],
        template=[("T", x.dtype), ("H", h), ("W", w), ("OH", size[0]), ("OW", size[1]), ("ALIGN", align)],
        grid=(size[1], size[0], n * c),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, c, size[0], size[1])],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip


_BILINEAR_NHWC = metal_kernel(
    name="graph_bilinear_nhwc",
    input_names=["x", "r"],
    output_names=["out"],
    source="""
        // `graph_bilinear` on an NHWC map: one thread per output channel value, channel-fastest (coalesced taps)
        uint ch = thread_position_in_grid.x, ox = thread_position_in_grid.y, z = thread_position_in_grid.z;
        if (ch >= C || ox >= OW) return;
        uint n = z / OH, oy = z % OH;
        float ys = ALIGN ? float(oy) * r[0] : max((float(oy) + 0.5f) * r[0] - 0.5f, 0.0f);
        float xs = ALIGN ? float(ox) * r[1] : max((float(ox) + 0.5f) * r[1] - 0.5f, 0.0f);
        int y0 = min(int(ys), H - 1), x0 = min(int(xs), W - 1);
        int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float wy = ys - float(y0), wx = xs - float(x0);
        const device T* p = x + size_t(n) * H * W * C + ch;
        float top = float(p[(y0 * W + x0) * C]) * (1.0f - wx) + float(p[(y0 * W + x1) * C]) * wx;
        float bot = float(p[(y1 * W + x0) * C]) * (1.0f - wx) + float(p[(y1 * W + x1) * C]) * wx;
        out[((size_t(n) * OH + oy) * OW + ox) * C + ch] = T(top * (1.0f - wy) + bot * wy);
    """,
)


def bilinear_nhwc(x: mx.array, size: tuple[int, int], ratios: tuple[float, float], *, align: bool) -> mx.array:
    """`bilinear` of an NHWC map (`[N, H, W, C]` in and out)."""
    n, h, w, c = x.shape
    tg = min(c, 32)
    return _BILINEAR_NHWC(
        inputs=[x, mx.array(ratios, dtype=mx.float32)],
        template=[("T", x.dtype), ("C", c), ("H", h), ("W", w), ("OH", size[0]), ("OW", size[1]), ("ALIGN", align)],
        grid=(-(-c // tg) * tg, size[1], n * size[0]),
        threadgroup=(tg, max(1, 256 // tg), 1),
        output_shapes=[(n, size[0], size[1], c)],
        output_dtypes=[x.dtype],
    )[0]  # fmt: skip


_TRILINEAR = metal_kernel(
    name="graph_trilinear",
    input_names=["x", "r"],
    output_names=["out"],
    source="""
        // one thread per output voxel of one plane: the eight taps, fp32 weights, one rounding (no intermediates)
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y, z = thread_position_in_grid.z;
        if (ox >= OW || oy >= OH) return;
        uint plane = z / OD, od = z % OD;
        float zs = ALIGN ? float(od) * r[0] : max((float(od) + 0.5f) * r[0] - 0.5f, 0.0f);
        float ys = ALIGN ? float(oy) * r[1] : max((float(oy) + 0.5f) * r[1] - 0.5f, 0.0f);
        float xs = ALIGN ? float(ox) * r[2] : max((float(ox) + 0.5f) * r[2] - 0.5f, 0.0f);
        int z0 = min(int(zs), D - 1), y0 = min(int(ys), H - 1), x0 = min(int(xs), W - 1);
        int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float wz = zs - float(z0), wy = ys - float(y0), wx = xs - float(x0);
        const device T* p = x + size_t(plane) * D * H * W;
        float acc = 0.0f;
        for (int dz = 0; dz < 2; ++dz) {
            int zz = dz ? z1 : z0;
            float fz = dz ? wz : 1.0f - wz;
            const device T* q = p + size_t(zz) * H * W;
            float top = float(q[y0 * W + x0]) * (1.0f - wx) + float(q[y0 * W + x1]) * wx;
            float bot = float(q[y1 * W + x0]) * (1.0f - wx) + float(q[y1 * W + x1]) * wx;
            acc += fz * (top * (1.0f - wy) + bot * wy);
        }
        out[(size_t(z) * OH + oy) * OW + ox] = T(acc);
    """,
)


def trilinear(x: mx.array, size: tuple[int, int, int], ratios: tuple[float, float, float], *, align: bool) -> mx.array:
    """Trilinear resize of an NCDHW volume (torch's `upsample_trilinear3d`); same conventions as `bilinear`."""
    n, c, d, h, w = x.shape
    return _TRILINEAR(
        inputs=[mx.contiguous(x), mx.array(ratios, dtype=mx.float32)],
        template=[("T", x.dtype), ("D", d), ("H", h), ("W", w), ("OD", size[0]), ("OH", size[1]), ("OW", size[2]),
                  ("ALIGN", align)],
        grid=(size[2], size[1], n * c * size[0]), threadgroup=(32, 8, 1),
        output_shapes=[(n, c, *size)], output_dtypes=[x.dtype],
    )[0]  # fmt: skip


_GWC = metal_kernel(
    name="graph_gwc_volume",
    input_names=["l", "r"],
    output_names=["out"],
    source="""
        // one thread per (pixel, disparity): G group means of left[x] * right[x - d] (NHWC features, [D, H, W, G] out)
        uint x = thread_position_in_grid.x, y = thread_position_in_grid.y, nd = thread_position_in_grid.z;
        if (x >= W || y >= H) return;
        uint n = nd / D, d = nd % D;
        const device T* lp = l + ((size_t(n) * H + y) * W + x) * C;
        const device T* rp = r + ((size_t(n) * H + y) * W + (x - d)) * C;
        device T* op = out + (((size_t(n) * D + d) * H + y) * W + x) * G;
        constexpr int CG = C / G;
        for (int g = 0; g < G; ++g) {
            float acc = 0.0f;
            if (x >= d) {
                for (int k = 0; k < CG; ++k) acc += float(lp[g * CG + k]) * float(rp[g * CG + k]);
            }
            op[g] = T(acc / float(CG));
        }
    """,
)


def gwc_volume(left: mx.array, right: mx.array, depth: int, groups: int) -> mx.array:
    """Group-wise correlation volume of two NHWC feature maps; returns `[B, D, H, W, G]` (zeros where `x < d`)."""
    t = mx.result_type(left.dtype, right.dtype)  # the kernel reads both as one type
    left, right = left.astype(t), right.astype(t)
    n, h, w, c = left.shape
    return _GWC(
        inputs=[mx.contiguous(left), mx.contiguous(right)],
        template=[("T", left.dtype), ("C", c), ("G", groups), ("W", w), ("H", h), ("D", depth)],
        grid=(w, h, n * depth), threadgroup=(32, 4, 1),
        output_shapes=[(n, depth, h, w, groups)], output_dtypes=[left.dtype],
    )[0]  # fmt: skip


_LINEAR_SAMPLE = metal_kernel(
    name="graph_linear_sample",
    input_names=["vol", "x"],
    output_names=["out"],
    source="""
        // one thread per (position, channel, tap): lerp of vol[p, c, :] at x[p, k], zeros outside (fp32 math)
        uint j = thread_position_in_grid.x, p = thread_position_in_grid.y;
        if (j >= C * K || p >= P) return;
        uint c = j / K, k = j % K;
        float xv = x[size_t(p) * K + k];
        float x0 = floor(xv), w1 = xv - x0;
        int i0 = int(x0);
        const device T* v = vol + (size_t(p) * C + c) * D;
        float acc = 0.0f;
        if (i0 >= 0 && i0 < D) acc += float(v[i0]) * (1.0f - w1);
        if (i0 + 1 >= 0 && i0 + 1 < D) acc += float(v[i0 + 1]) * w1;
        out[size_t(p) * C * K + j] = acc;
    """,
)


def linear_sample(vol: mx.array, x: mx.array) -> mx.array:
    """`fastforks::linear_sample`: `vol` `[P, C, D]` or `[P, D]`, `x` `[P, K]` -> `[P, C * K]` float32."""
    p, d = vol.shape[0], vol.shape[-1]
    c = vol.shape[1] if vol.ndim == 3 else 1
    k = x.shape[1]
    tx = min(c * k, 64)
    return _LINEAR_SAMPLE(
        inputs=[mx.contiguous(vol), mx.contiguous(x.astype(mx.float32))],
        template=[("T", vol.dtype), ("P", p), ("C", c), ("K", k), ("D", d)],
        grid=(c * k, p, 1), threadgroup=(tx, max(1, 256 // tx), 1),
        output_shapes=[(p, c * k)], output_dtypes=[mx.float32],
    )[0]  # fmt: skip


_GWC_WARP = metal_kernel(
    name="graph_gwc_warp",
    input_names=["l", "r", "disp"],
    output_names=["out"],
    source="""
        // one thread per (pixel, sample): the four bilinear taps of `r` at (x - disp, y) (upstream's grid_sample
        // coordinates: normalized by W - 1, align_corners=False, zeros padding) dotted with `l` at the pixel, per
        // group, fp32 sums; NHWC features, `[B, D, H, W]` disparities, `[B, G, D, H, W]` out, zero where x < disp.
        uint x = thread_position_in_grid.x, y = thread_position_in_grid.y, bd = thread_position_in_grid.z;
        if (x >= W || y >= H) return;
        uint b = bd / D, dd = bd % D;
        float dv = disp[(size_t(bd) * H + y) * W + x];
        size_t o = ((size_t(b) * G * D + dd) * H + y) * W + x;
        if (float(x) < dv) {
            for (uint g = 0; g < G; ++g) out[o + size_t(g) * D * H * W] = T(0.0f);
            return;
        }
        float gx = (float(x) - dv) / ((float(W) - 1.0f) / 2.0f) - 1.0f;
        float gy = float(y) / ((float(H) - 1.0f) / 2.0f) - 1.0f;
        float ix = ((gx + 1.0f) * float(W) - 1.0f) / 2.0f, iy = ((gy + 1.0f) * float(H) - 1.0f) / 2.0f;
        float fx = floor(ix), fy = floor(iy);
        float ax = ix - fx, ay = iy - fy;
        int x0 = int(fx), y0 = int(fy);
        float wt[4];
        size_t base[4];
        for (int t = 0; t < 4; ++t) {
            int yy = y0 + (t >> 1), xx = x0 + (t & 1);
            bool ok = yy >= 0 && yy < int(H) && xx >= 0 && xx < int(W);
            wt[t] = ok ? ((t >> 1) ? ay : 1.0f - ay) * ((t & 1) ? ax : 1.0f - ax) : 0.0f;
            base[t] = ok ? ((size_t(b) * H + yy) * W + xx) * C : 0;
        }
        const device T* lp = l + ((size_t(b) * H + y) * W + x) * C;
        for (uint g = 0; g < G; ++g) {
            float acc = 0.0f;
            for (uint k = g * CG; k < (g + 1) * CG; ++k) {
                float s = float(r[base[0] + k]) * wt[0] + float(r[base[1] + k]) * wt[1]
                        + float(r[base[2] + k]) * wt[2] + float(r[base[3] + k]) * wt[3];
                acc += float(lp[k]) * s;
            }
            out[o + size_t(g) * D * H * W] = T(acc / float(CG));
        }
    """,
)


def gwc_warp(left: mx.array, right: mx.array, disp: mx.array, groups: int) -> mx.array:
    """`fastforks::gwc_warp` of NHWC `left`, `right` `[B, H, W, C]`, `[B, D, H, W]` disparities: `[B, G, D, H, W]`."""
    t = mx.result_type(left.dtype, right.dtype)  # the kernel reads both as one type
    b, h, w, c = left.shape
    d = disp.shape[1]
    return _GWC_WARP(
        inputs=[mx.contiguous(left.astype(t)), mx.contiguous(right.astype(t)), mx.contiguous(disp.astype(mx.float32))],
        template=[("T", t), ("C", c), ("CG", c // groups), ("G", groups), ("H", h), ("W", w), ("D", d)],
        grid=(w, h, b * d), threadgroup=(32, 8, 1),
        output_shapes=[(b, groups, d, h, w)], output_dtypes=[t],
    )[0]  # fmt: skip


_WARP_DOT = metal_kernel(
    name="graph_warp_dot",
    input_names=["f", "o", "grid"],
    output_names=["out"],
    source="""
        // one SIMD group per (view, depth, pixel): each lane takes VEC-channel slices of the four bilinear taps of
        // `o` and of `f` at the pixel (NHWC, coalesced rows), fp32 products, one simd_sum. No warped volume.
        uint lane = thread_position_in_grid.x, pix = thread_position_in_grid.y, nd = thread_position_in_grid.z;
        uint n = nd / D;
        bool live = pix < H * W;
        float gx = 0.0f, gy = 0.0f;
        if (live) {
            size_t gi = (size_t(nd) * H * W + pix) * 2;
            gx = grid[gi];
            gy = grid[gi + 1];
        }
        float ix = (gx + 1.0f) * float(W - 1) * 0.5f, iy = (gy + 1.0f) * float(H - 1) * 0.5f;
        float fx = floor(ix), fy = floor(iy);
        float ax = ix - fx, ay = iy - fy;
        int x0 = int(fx), y0 = int(fy);
        const device vec<T, VEC>* o4 = (const device vec<T, VEC>*)o;
        const device vec<T, VEC>* f4 = (const device vec<T, VEC>*)f;
        float acc = 0.0f;
        if (live) {
            for (uint cv = lane; cv < C / VEC; cv += 32) {
                float4 s = float4(0.0f);
                for (int dy = 0; dy < 2; ++dy) {
                    int yy = y0 + dy;
                    if (yy < 0 || yy >= H) continue;
                    float wy = dy ? ay : 1.0f - ay;
                    for (int dx = 0; dx < 2; ++dx) {
                        int xx = x0 + dx;
                        if (xx < 0 || xx >= W) continue;
                        float wgt = wy * (dx ? ax : 1.0f - ax);
                        vec<T, VEC> v = o4[((size_t(n) * H + yy) * W + xx) * (C / VEC) + cv];
                        for (int k = 0; k < VEC; ++k) s[k] += float(v[k]) * wgt;
                    }
                }
                vec<T, VEC> a = f4[(size_t(n) * H * W + pix) * (C / VEC) + cv];
                for (int k = 0; k < VEC; ++k) acc += float(a[k]) * s[k];
            }
        }
        acc = simd_sum(acc);
        if (live && lane == 0) out[size_t(nd) * H * W + pix] = acc;
    """,
)


def warp_dot(feat: mx.array, other: mx.array, grid: mx.array) -> mx.array:
    """`fastforks::warp_dot` of NHWC `feat`, `other` `[N, H, W, C]` and a `[N, D, H*W, 2]` grid; `[N, D, H, W]` fp32."""
    t = mx.result_type(feat.dtype, other.dtype)  # the kernel reads both as one type
    n, h, w, c = feat.shape
    d = grid.shape[1]
    vec = 4 if c % 4 == 0 else 2 if c % 2 == 0 else 1
    return _WARP_DOT(
        inputs=[mx.contiguous(feat.astype(t)), mx.contiguous(other.astype(t)), mx.contiguous(grid.astype(mx.float32))],
        template=[("T", t), ("C", c), ("H", h), ("W", w), ("D", d), ("VEC", vec)],
        grid=(32, -(-(h * w) // 8) * 8, n * d), threadgroup=(32, 8, 1),
        output_shapes=[(n, d, h, w)], output_dtypes=[mx.float32],
    )[0]  # fmt: skip


_USA = metal_kernel(
    name="graph_upsample_soft_argmax",
    input_names=["x"],
    output_names=["out"],
    source="""
        // one thread per output pixel: trilinear taps of the low-resolution volume, online softmax over the MD
        // disparities, expected disparity. The full-resolution volume never exists; fp32 throughout.
        uint ox = thread_position_in_grid.x, oy = thread_position_in_grid.y, n = thread_position_in_grid.z;
        if (ox >= OW || oy >= OH) return;
        float ys = max((float(oy) + 0.5f) / float(SH) - 0.5f, 0.0f);
        float xs = max((float(ox) + 0.5f) / float(SW) - 0.5f, 0.0f);
        int y0 = min(int(ys), H - 1), x0 = min(int(xs), W - 1);
        int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float wy = ys - float(y0), wx = xs - float(x0);
        const device T* p = x + size_t(n) * D * H * W;
        int o00 = y0 * W + x0, o01 = y0 * W + x1, o10 = y1 * W + x0, o11 = y1 * W + x1;
        float m = -INFINITY, s = 0.0f, t = 0.0f;
        int cur = -1;
        float a0 = 0.0f, a1 = 0.0f;
        for (int j = 0; j < MD; ++j) {
            float zs = max((float(j) + 0.5f) / float(SD) - 0.5f, 0.0f);
            int z0 = min(int(zs), D - 1), z1 = min(z0 + 1, D - 1);
            if (z0 != cur) {
                const device T* q0 = p + size_t(z0) * H * W;
                const device T* q1 = p + size_t(z1) * H * W;
                a0 = (float(q0[o00]) * (1.0f - wx) + float(q0[o01]) * wx) * (1.0f - wy)
                   + (float(q0[o10]) * (1.0f - wx) + float(q0[o11]) * wx) * wy;
                a1 = (float(q1[o00]) * (1.0f - wx) + float(q1[o01]) * wx) * (1.0f - wy)
                   + (float(q1[o10]) * (1.0f - wx) + float(q1[o11]) * wx) * wy;
                cur = z0;
            }
            float wz = zs - float(z0);
            float v = a0 * (1.0f - wz) + a1 * wz;
            if (v > m) {
                float k = exp(m - v);
                s *= k;
                t *= k;
                m = v;
            }
            float e = exp(v - m);
            s += e;
            t += e * float(j);
        }
        out[(size_t(n) * OH + oy) * OW + ox] = t / s;
    """,
)


def upsample_soft_argmax(cost: mx.array, max_disp: int, height: int, width: int) -> mx.array:
    """Soft-argmin disparity `[B, H, W]` (float32) of a `[B, 1, d, h, w]` cost volume, trilinearly upsampled."""
    n, _, d, h, w = cost.shape
    return _USA(
        inputs=[mx.contiguous(cost)],
        template=[("T", cost.dtype), ("D", d), ("H", h), ("W", w), ("MD", max_disp), ("OH", height), ("OW", width),
                  ("SD", max_disp // d), ("SH", height // h), ("SW", width // w)],
        grid=(width, height, n), threadgroup=(32, 4, 1),
        output_shapes=[(n, height, width)], output_dtypes=[mx.float32],
    )[0]  # fmt: skip


_WIN_SRC = """
        // one threadgroup per (window, head), one simdgroup per QT * 8 queries (QT = 2 is fastest at L = 64); K and
        // V of the window staged in threadgroup memory (faster than reading them through the cache). S = Q K^T by
        // 8x8 simdgroup MMA with float32 accumulators kept in registers; scale, additive masks and the row softmax
        // in float32; P rounded to T for O = P V by MMA.
        // MLX's 8x8 fragment layout: a lane holds row fm, columns fn and fn + 1; the lanes of a row differ in lane
        // bits 0 and 3, so row reductions are xor-shuffles over 1 and 8.
        uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
        uint bh = threadgroup_position_in_grid.x;
        uint b = bh / NH, h = bh % NH;
        // q, k and v rows `RS` apart from `b * BS + h * HS` (plus `KO` / `VO`): `[B, H, L, D]` arrays, or the one
        // `[B, L, 3, H, D]` projection a fused qkv linear gives, read where it lies
        size_t base = size_t(b) * BS + size_t(h) * HS;
        threadgroup T ks[L * D];
        threadgroup T vs[L * D];
        {
            const device vec<T, 4>* kp = (const device vec<T, 4>*)(k + base + KO);
            const device vec<T, 4>* vp = (const device vec<T, 4>*)(v + base + VO);
            for (uint e = thread_position_in_threadgroup.x; e < uint(L * D / 4); e += TPG) {
                uint src = (e / (D / 4)) * (RS / 4) + e % (D / 4);
                ((threadgroup vec<T, 4>*)ks)[e] = kp[src];
                ((threadgroup vec<T, 4>*)vs)[e] = vp[src];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        short qid = lane / 4;
        short fm = (qid & 4) + ((lane / 2) % 4);
        short fn = (qid & 2) * 2 + (lane % 2) * 2;
        float scale = sc[0];
        simdgroup_float8x8 sm[QT][L / 8];
        float mx_[QT], inv[QT];
        for (int qt = 0; qt < QT; ++qt) {
            uint r0 = (sg * QT + qt) * 8;
            const device T* qp = q + base + size_t(r0) * RS;
            simdgroup_matrix<T, 8, 8> qm[D / 8];
            for (int dt = 0; dt < D / 8; ++dt) simdgroup_load(qm[dt], qp + dt * 8, RS);
            const device M1* r1 = m1 + ((size_t(b % MB1) * MH1 + (MH1 > 1 ? h : 0)) * L + r0 + fm) * L;
            const device M2* r2 = m2 + ((size_t(b % MB2) * MH2 + (MH2 > 1 ? h : 0)) * L + r0 + fm) * L;
            float m = -INFINITY;
            for (int jt = 0; jt < L / 8; ++jt) {
                simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
                for (int dt = 0; dt < D / 8; ++dt) {
                    simdgroup_matrix<T, 8, 8> kt;  // K^T tile: (d, key)
                    simdgroup_load(kt, ks + jt * 8 * D + dt * 8, D, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc, qm[dt], kt, acc);
                }
                for (int e = 0; e < 2; ++e) {
                    float x = acc.thread_elements()[e] * scale;
                    if (HAS1) x += float(r1[jt * 8 + fn + e]);
                    if (HAS2) x += float(r2[jt * 8 + fn + e]);
                    acc.thread_elements()[e] = x;
                    m = max(m, x);
                }
                sm[qt][jt] = acc;
            }
            m = max(m, simd_shuffle_xor(m, 1));
            m = max(m, simd_shuffle_xor(m, 8));
            float sum = 0.0f;
            for (int jt = 0; jt < L / 8; ++jt) {
                for (int e = 0; e < 2; ++e) {
                    float x = exp(sm[qt][jt].thread_elements()[e] - m);
                    sm[qt][jt].thread_elements()[e] = x;
                    sum += x;
                }
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 8);
            inv[qt] = 1.0f / sum;
        }
        for (int dt = 0; dt < D / 8; ++dt) {
            simdgroup_float8x8 acc[QT];
            for (int qt = 0; qt < QT; ++qt) acc[qt] = simdgroup_float8x8(0.0f);
            for (int jt = 0; jt < L / 8; ++jt) {
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(vm, vs + jt * 8 * D + dt * 8, D);
                for (int qt = 0; qt < QT; ++qt) {
                    simdgroup_matrix<T, 8, 8> pm;
                    pm.thread_elements()[0] = T(sm[qt][jt].thread_elements()[0]);
                    pm.thread_elements()[1] = T(sm[qt][jt].thread_elements()[1]);
                    simdgroup_multiply_accumulate(acc[qt], pm, vm, acc[qt]);
                }
            }
            for (int qt = 0; qt < QT; ++qt) {
                // out is [B, L, H, D]
                device T* op = out + ((size_t(b) * L + (sg * QT + qt) * 8 + fm) * NH + h) * D + dt * 8 + fn;
                op[0] = T(acc[qt].thread_elements()[0] * inv[qt]);
                op[1] = T(acc[qt].thread_elements()[1] * inv[qt]);
            }
        }
"""


_WIN_ATTN = metal_kernel(
    name="graph_window_attention",
    input_names=["q", "k", "v", "m1", "m2", "sc"],
    output_names=["out"],
    header="#include <metal_simdgroup_matrix>\n",
    source=_WIN_SRC,
)

WIN_ATTN_TG_BYTES = 32768
"""Threadgroup memory of `window_attention`: K and V of one window and head in the input dtype."""


def window_attention_ok(q: mx.array, k: mx.array, v: mx.array) -> bool:
    """Whether `window_attention` runs these shapes (self-attention, `L` and `D` multiples of 8, fits 32 KB)."""
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape or q.dtype not in (mx.float16, mx.float32):
        return False
    n, d = q.shape[2:]
    return n % 8 == 0 and d % 8 == 0 and n <= 256 and d <= 128 and 2 * n * d * q.dtype.size <= WIN_ATTN_TG_BYTES


def window_attention(
    q: mx.array, k: mx.array | None, v: mx.array | None, scale: float, masks: list[mx.array]
) -> mx.array:
    """`softmax(q k^T * scale + sum(masks)) v` of short self-attention (`[B, H, L, D]`, see `window_attention_ok`).

    With `k` and `v` None, `q` is the `[B, L, 3, H, D]` output of a fused qkv projection, read in place (slicing q, k
    and v out of it costs MLX a strided copy each). Each of the (at most two) additive masks is `[MB, MH, L, L]` with
    `MB` dividing `B` (batch `b` reads mask `b % MB`, as a broadcast or a `repeat` over the batch gives) and `MH` 1 or
    `H`; the summed `[B, H, L, L]` mask is never built. Scores and softmax in float32. The result is a `[B, H, L, D]`
    view of a `[B, L, H, D]` array, so the usual `transpose(1, 2).reshape(B, L, H * D)` after it is free.
    """
    if k is None or v is None:
        qkv = mx.contiguous(q)
        b, n, _, nh, d = qkv.shape
        q = k = v = qkv
        strides = [("RS", 3 * nh * d), ("BS", n * 3 * nh * d), ("HS", d), ("KO", nh * d), ("VO", 2 * nh * d)]
    else:
        b, nh, n, d = q.shape
        q, k, v = mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)
        strides = [("RS", d), ("BS", nh * n * d), ("HS", n * d), ("KO", 0), ("VO", 0)]
    dummy = mx.zeros((1, 1, n, n), q.dtype)  # a one-element array would be bound as `constant`
    ms = [mx.contiguous(m) for m in masks] + [dummy] * (2 - len(masks))
    qt = 2 if n % 16 == 0 else 1
    tpg = n // qt * 4  # one simdgroup per QT * 8 queries
    tmpl = [("T", q.dtype), ("NH", nh), ("L", n), ("D", d), ("TPG", tpg), ("QT", qt), *strides]
    for i, m in enumerate(ms, 1):
        tmpl += [(f"M{i}", m.dtype), (f"MB{i}", m.shape[0]), (f"MH{i}", m.shape[1]), (f"HAS{i}", i <= len(masks))]
    return _WIN_ATTN(
        inputs=[q, k, v, *ms, mx.array([scale], mx.float32)],
        template=tmpl,
        grid=(tpg * b * nh, 1, 1), threadgroup=(tpg, 1, 1),
        output_shapes=[(b, n, nh, d)], output_dtypes=[q.dtype],
    )[0].swapaxes(1, 2)  # fmt: skip


_ROW_LN = metal_kernel(
    name="graph_row_layer_norm",
    input_names=["x", "w", "b", "eps"],
    output_names=["out"],
    source="""
        // one simdgroup per row of C (<= 32 * NPL) elements held in registers: float32 mean, then the centred variance
        uint lane = thread_index_in_simdgroup;
        size_t row = size_t(threadgroup_position_in_grid.x) * SG + simdgroup_index_in_threadgroup;
        if (row >= size_t(ROWS)) return;
        const device T* xr = x + row * C;
        float v[NPL];
        float s = 0.0f;
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            v[i] = c < uint(C) ? float(xr[c]) : 0.0f;
            s += v[i];
        }
        float mean = simd_sum(s) / C;
        float q = 0.0f;
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            float d = c < uint(C) ? v[i] - mean : 0.0f;
            q += d * d;
        }
        float rstd = rsqrt(simd_sum(q) / C + eps[0]);
        device T* o = out + row * C;
        for (int i = 0; i < NPL; ++i) {
            uint c = lane + 32 * i;
            if (c < uint(C)) {
                float y = (v[i] - mean) * rstd;
                if (HAS_W) y *= float(w[c]);
                if (HAS_B) y += float(b[c]);
                o[c] = T(y);
            }
        }
    """,
)

ROW_LN_MAX = 512
"""Widest row `row_layer_norm` takes (MLX 0.32, M1 Pro, 47M elements: never slower up to 512, equal at 240, 256 and
512; at 768 `mx.fast.layer_norm` is 15 % faster)."""


def row_layer_norm(x: mx.array, w: mx.array | None, b: mx.array | None, eps: float) -> mx.array:
    """Layer norm over the last axis for narrow rows (`C <= ROW_LN_MAX`): one simdgroup per row.

    `mx.fast.layer_norm` is several times slower at many such widths (47M float16 elements: C = 180 8.1 against 1.3
    ms, C = 384 5.8 against 1.3 ms, C = 96 25 against 1.3 ms). `w` and `b` are read in `x`'s dtype.
    """
    c = x.shape[-1]
    rows = x.size // c
    sg = 8
    dummy = mx.zeros((c,), x.dtype)
    return _ROW_LN(
        inputs=[mx.contiguous(x), w if w is not None else dummy, b if b is not None else dummy,
                mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("C", c), ("NPL", -(-c // 32)), ("SG", sg), ("ROWS", rows),
                  ("HAS_W", w is not None), ("HAS_B", b is not None)],
        grid=(-(-rows // sg) * 32 * sg, 1, 1), threadgroup=(32 * sg, 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]  # fmt: skip


def _corr_windows_kernel(levels: int) -> Callable[..., list[mx.array]]:
    """The `corr_windows` kernel of a pyramid of `levels` levels (one input array per level, no concatenated copy)."""
    if levels not in _CORR_WINDOWS:
        pick = "\n".join(f"if (l == {i}) {{ f2 = f2_{i}; hl = H{i}; wl = W{i}; }}" for i in range(levels))
        _CORR_WINDOWS[levels] = metal_kernel(
            name=f"graph_corr_windows_l{levels}",
            input_names=["f1", *(f"f2_{i}" for i in range(levels)), "coords", "sc"],
            output_names=["out"],
            source=_CORR_WINDOWS_SRC.replace("PICK_LEVEL", pick),
        )
    return _CORR_WINDOWS[levels]


_CORR_WINDOWS: dict[int, Callable[..., list[mx.array]]] = {}
_CORR_WINDOWS_SRC = """
        // RAFT's correlation windows without a volume (`flow_metal`'s corr_window, the MPS kernel, for MLX). One SIMD
        // group per (pixel, level): the (2r + 2)^2 integer neighbours' dot products, LANES lanes per neighbour (f1's
        // C / LANES channels of the lane kept in registers), G = 32 / LANES neighbours at a time; then the bilinear
        // (2r + 1)^2 window (align_corners, zeros outside). fp32 dot products and output.
        constexpr uint G = 32 / LANES, S = 2 * R + 2, K = 2 * R + 1, NV = C / (4 * LANES);
        threadgroup float dots[4][S * S + G];
        uint gid = thread_position_in_grid.x;
        uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
        uint item = gid / 32;  // (b * P + n) * L + l
        if (item >= NB * P * L) return;  // whole SIMD groups only
        uint l = item % L, bn = item / L, b = bn / P, n = bn - b * P;
        uint sub = lane / LANES, s = lane % LANES;
        float4 a[NV];
        const device T* p1 = f1 + size_t(bn) * C + s * 4;
        for (uint v = 0; v < NV; ++v) a[v] = float4(*(const device vec<T, 4>*)(p1 + v * 4 * LANES));
        float lsc = 1.0f / float(1u << l);
        float cx = float(coords[size_t(b) * 2 * P + n]) * lsc, cy = float(coords[size_t(b) * 2 * P + P + n]) * lsc;
        bool ok = isfinite(cx) && isfinite(cy) && fabs(cx) < 1.0e6f && fabs(cy) < 1.0e6f;
        if (!ok) cx = cy = -1.0e7f;  // every tap outside: zeros
        float x0 = floor(cx), y0 = floor(cy), ax = cx - x0, ay = cy - y0;
        const device T* f2 = f2_0;
        int hl = H0, wl = W0;
        PICK_LEVEL
        int ix = int(clamp(x0, -2.0e6f, 2.0e6f)) - R, iy = int(clamp(y0, -2.0e6f, 2.0e6f)) - R;
        const device T* base = f2 + size_t(b) * hl * wl * C + s * 4;
        float scale = sc[0];
        for (uint pos = sub; pos < S * S + sub; pos += G) {  // the same pass count on every lane (the shuffles)
            int x = ix + int(pos / S), y = iy + int(pos % S);
            bool in = pos < S * S && x >= 0 && x < wl && y >= 0 && y < hl;
            const device T* p2 = base + (size_t(clamp(y, 0, hl - 1)) * wl + clamp(x, 0, wl - 1)) * C;
            float t = 0.0f;
            for (uint v = 0; v < NV; ++v) t += dot(a[v], float4(*(const device vec<T, 4>*)(p2 + v * 4 * LANES)));
            for (uint m = LANES / 2; m > 0; m /= 2) t += simd_shuffle_xor(t, m);
            if (s == 0 && pos < S * S) dots[sg][pos] = in ? t * scale : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float w00 = (1.0f - ax) * (1.0f - ay), w10 = ax * (1.0f - ay), w01 = (1.0f - ax) * ay, w11 = ax * ay;
        device float* o = out + (size_t(b) * L + l) * K * K * P + n;
        for (uint k = lane; k < K * K; k += 32) {
            uint i = k / K, j = k - i * K, d = i * S + j;
            o[size_t(k) * P] = w00 * dots[sg][d] + w10 * dots[sg][d + S] + w01 * dots[sg][d + 1]
                + w11 * dots[sg][d + S + 1];
        }
"""


def corr_windows(f1: mx.array, levels: list[mx.array], coords: mx.array, radius: int, scale: float) -> mx.array:
    """`fastforks::corr_windows` of NHWC `f1` `[B, H, W, C]`, NHWC `levels`, `coords` `[B, 2, H, W]`: fp32 windows.

    Returns `[B, L * (2r + 1)^2, H, W]` (see `fastforks._core.volume_ops.corr_windows`).
    """
    t = f1.dtype
    for f in levels:
        t = mx.result_type(t, f.dtype)
    b, h, w, c = f1.shape
    lanes = 8 if c <= 512 else 32  # f1 in registers: C / LANES floats per lane
    if c % (4 * lanes):
        msg = f"corr_windows: {c} channels (a multiple of {4 * lanes})"
        raise ValueError(msg)
    nl, p = len(levels), h * w
    tmpl = [("T", t), ("C", c), ("R", radius), ("L", nl), ("P", p), ("NB", b), ("LANES", lanes)]
    for i, f in enumerate(levels):
        tmpl += [(f"H{i}", f.shape[1]), (f"W{i}", f.shape[2])]
    return _corr_windows_kernel(nl)(
        inputs=[mx.contiguous(f1.astype(t)), *(mx.contiguous(f.astype(t)) for f in levels),
                mx.contiguous(coords.astype(mx.float32)), mx.array([scale, 0.0], mx.float32)],
        template=tmpl,
        grid=(b * p * nl * 32, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(b, nl * (2 * radius + 1) ** 2, h, w)], output_dtypes=[mx.float32],
    )[0]  # fmt: skip
