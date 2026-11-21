"""Channels-last depthwise convolution and global pooling on Apple MLX (`fastforks.shared.mlx_graph`, kernels mode).

MLX's grouped convolution runs a depthwise layer as many tiny convolutions; one thread per output element with the
window held in registers is several times faster for the 3x3 .. 7x7 kernels of mobile and ConvNeXt-style networks.
Accumulation is float32, the result is rounded once to the activation dtype.
"""

from __future__ import annotations

from fastforks._core.mlx import metal_kernel, mx

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
