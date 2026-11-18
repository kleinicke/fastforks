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
    """NHWC depthwise convolution (channels a multiple of 4). `w` is `[kh, kw, C]`, `b` is `[C]`."""
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
            ("PH", padding[0]), ("PW", padding[1]), ("DH", dilation[0]), ("DW", dilation[1]), ("OHT", oh), ("OWT", ow), ("R", r),
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
