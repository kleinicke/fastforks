"""CREStereo on Apple MLX, kernels mode: Metal kernels (`mx.fast.metal_kernel`) for everything around the convolutions.

The same operations as `stereo/models/crestereo/mlx_ops.py` (the portable ones), under the same names:
`instance_norm_relu`, `corr_iter`, `corr_offset`, `gru`, `convex_upsample`, plus `extend` (extra parameters).

- `agcl_warp` + `agcl_window`: the iterative group correlation (warp the right features by the flow once, then the
  3x3 / 1x9 window dot products); `agcl_offset`: the correlation at learned offsets (bilinear taps and group dot
  products in one kernel). Features are read in their storage dtype (fp16 or fp32), every interpolation and dot
  product accumulates in float32.
- `gru_pack` / `gru_gates` / `gru_update`: the GRU epilogues. Each also writes the next convolution's input
  (`[hidden | motion | flow]`, `[r * hidden | motion | flow]`), so no concatenation or slice is ever materialised.
- `wino_in` / `gru_gates_wino` / `gru_update_wino`: the GRU's 1x5 / 5x1 convolutions as 1-D Winograd F(4, 5): 8
  multiplications per 4 outputs instead of 20. `wino_in` writes the transformed input tiles `[8, tiles, C]`, one
  float32 matmul with the transformed weights (`extend`) does the multiplications, and the output transform is
  folded into the gate / update epilogues. Always float32 (the transform's constants would cost float16 too much
  precision; the float32 matmul is as fast). With the tunable `winograd` off, MLX's convolution runs instead.
- `inorm_stats` / `inorm_apply`: instance norm with float32 statistics (two reads of the map instead of six).
- `convex_up`: softmax over the 9 mask logits and the weighted 3x3 flow sum, written at 4x.

All of this is Metal 3-level MSL that every Apple-silicon GPU runs (no simdgroup matrices, no tensor types).
Threadgroup size, warp lane and the Winograd switch come from the tunables table (`fastforks._core.mlx.TUNING`).
"""
# ruff: noqa: ANN001, ANN201, ANN202, D103, PLR0913, PLR0917, PLR2004, E501 - array-in / array-out functions
# named after the portable ops they replace (positional arrays); Metal source lines are kept whole

from __future__ import annotations

import numpy as np

from fastforks._core.mlx import metal_kernel, mx, tuning

GROUPS = 4  # correlation groups (upstream AGCL)
SEARCH = 9  # window positions: 3x3 or 1x9
HDIM = 128  # GRU hidden channels


def _tg() -> tuple[int, int, int]:
    return (tuning().threadgroup, 1, 1)


_HEADER = """
// bilinear taps of torch grid_sample(align_corners=True, padding_mode="zeros") at pixel (x, y)
struct Taps { uint i00, i01, i10, i11; float w00, w01, w10, w11; };
inline Taps taps(float x, float y, int W, int H) {
    x = clamp(x, -2.0f, float(W) + 1.0f);  // anything further out has zero weight anyway (and NaN stays finite)
    y = clamp(y, -2.0f, float(H) + 1.0f);
    float x0f = floor(x), y0f = floor(y);
    float fx = x - x0f, fy = y - y0f;
    int x0 = int(x0f), y0 = int(y0f), x1 = x0 + 1, y1 = y0 + 1;
    float ax0 = (x0 >= 0 && x0 < W) ? 1.0f - fx : 0.0f, ax1 = (x1 >= 0 && x1 < W) ? fx : 0.0f;
    float ay0 = (y0 >= 0 && y0 < H) ? 1.0f - fy : 0.0f, ay1 = (y1 >= 0 && y1 < H) ? fy : 0.0f;
    x0 = clamp(x0, 0, W - 1); x1 = clamp(x1, 0, W - 1); y0 = clamp(y0, 0, H - 1); y1 = clamp(y1, 0, H - 1);
    Taps t;
    t.i00 = uint(y0 * W + x0); t.i01 = uint(y0 * W + x1); t.i10 = uint(y1 * W + x0); t.i11 = uint(y1 * W + x1);
    t.w00 = ax0 * ay0; t.w01 = ax1 * ay0; t.w10 = ax0 * ay1; t.w11 = ax1 * ay1;
    return t;
}
"""

# one thread per (pixel, lane of L channels)
_WARP = metal_kernel(
    name="agcl_warp",
    input_names=["f2", "flow"],
    output_names=["out"],
    header=_HEADER,
    source="""
        int H = f2_shape[1], W = f2_shape[2];
        uint C = f2_shape[3], lanes = C / L;
        uint tid = thread_position_in_grid.x, lane = tid % lanes, pix = tid / lanes;
        int px = int(pix % uint(W)), py = int(pix / uint(W));
        Taps t = taps(float(px) + flow[2 * pix], float(py) + flow[2 * pix + 1], W, H);
        uint c0 = lane * L, o = pix * C + c0;
        uint a = t.i00 * C + c0, b = t.i01 * C + c0, c = t.i10 * C + c0, d = t.i11 * C + c0;
        for (uint k = 0; k < L; ++k)
            out[o + k] = T(t.w00 * float(f2[a + k]) + t.w01 * float(f2[b + k]) + t.w10 * float(f2[c + k])
                + t.w11 * float(f2[d + k]));
    """,
)

# one thread per (pixel, window position): the 4 group dot products with the (edge-clamped) warped neighbour
_WINDOW = metal_kernel(
    name="agcl_window",
    input_names=["f1", "wp"],
    output_names=["out"],
    source="""
        int H = f1_shape[1], W = f1_shape[2];
        uint C = f1_shape[3], G = C / 4;
        uint tid = thread_position_in_grid.x, o = tid % 9, pix = tid / 9;
        int px = int(pix % uint(W)), py = int(pix / uint(W));
        int dx = SMALL ? int(o % 3) - 1 : int(o) - 4, dy = SMALL ? int(o / 3) - 1 : 0;
        uint q = uint(clamp(py + dy, 0, H - 1) * W + clamp(px + dx, 0, W - 1)) * C, p = pix * C;
        for (uint g = 0; g < 4; ++g) {
            float4 acc = 0.0f;
            for (uint k = g * G; k < (g + 1) * G; k += 4)
                acc += float4(f1[p + k], f1[p + k + 1], f1[p + k + 2], f1[p + k + 3])
                    * float4(wp[q + k], wp[q + k + 1], wp[q + k + 2], wp[q + k + 3]);
            out[pix * 36 + g * 9 + o] = T((acc.x + acc.y + acc.z + acc.w) / float(G));
        }
    """,
)

# one thread per (pixel, window position): bilinear taps at the learned offset, 4 group dot products
_OFFSET = metal_kernel(
    name="agcl_offset",
    input_names=["f1", "f2", "flow", "offs"],
    output_names=["out"],
    header=_HEADER,
    source="""
        int H = f1_shape[1], W = f1_shape[2];
        uint C = f1_shape[3], G = C / 4;
        uint tid = thread_position_in_grid.x, o = tid % 9, pix = tid / 9;
        int px = int(pix % uint(W)), py = int(pix / uint(W));
        float dx = SMALL ? float(int(o % 3) - 1) : float(int(o) - 4), dy = SMALL ? float(int(o / 3) - 1) : 0.0f;
        Taps t = taps(float(px) + flow[2 * pix] + dx + offs[pix * 18 + 2 * o],
                      float(py) + flow[2 * pix + 1] + dy + offs[pix * 18 + 2 * o + 1], W, H);
        uint p = pix * C, a = t.i00 * C, b = t.i01 * C, c = t.i10 * C, d = t.i11 * C;
        for (uint g = 0; g < 4; ++g) {
            float acc = 0.0f;
            for (uint k = g * G; k < (g + 1) * G; ++k)
                acc += float(f1[p + k]) * (t.w00 * float(f2[a + k]) + t.w01 * float(f2[b + k])
                    + t.w10 * float(f2[c + k]) + t.w11 * float(f2[d + k]));
            out[pix * 36 + g * 9 + o] = T(acc / float(G));
        }
    """,
)


def corr_iter(f1, f2, flow, small):
    n, h, w, c = f1.shape
    lane = tuning().warp_lane
    warped = _WARP(inputs=[f2, flow], template=[("T", f2.dtype), ("L", lane)], grid=(h * w * (c // lane), 1, 1),
                   threadgroup=_tg(), output_shapes=[f2.shape], output_dtypes=[f2.dtype])[0]  # fmt: skip
    return _WINDOW(inputs=[f1, warped], template=[("T", f1.dtype), ("SMALL", bool(small))],
                   grid=(h * w * SEARCH, 1, 1), threadgroup=_tg(), output_shapes=[(n, h, w, GROUPS * SEARCH)],
                   output_dtypes=[f1.dtype])[0]  # fmt: skip


def corr_offset(f1, f2, offs, flow, small):
    n, h, w, _ = f1.shape
    out_dtype = mx.float32 if f1.dtype == mx.float32 else mx.float16
    return _OFFSET(inputs=[f1, f2, flow, offs], template=[("T", out_dtype), ("SMALL", bool(small))],
                   grid=(h * w * SEARCH, 1, 1), threadgroup=_tg(), output_shapes=[(n, h, w, GROUPS * SEARCH)],
                   output_dtypes=[out_dtype])[0]  # fmt: skip


# ----------------------------------------------------------------------------- GRU epilogues
# hx = [hidden (128) | relu(motion conv + bias) (126) | flow (2)]; one thread per element of hx
_PACK = metal_kernel(
    name="gru_pack",
    input_names=["hidden", "mconv", "mbias", "flow"],
    output_names=["hx"],
    source="""
        uint tid = thread_position_in_grid.x, c = tid % 256, pix = tid / 256;
        if (c < 128) hx[tid] = hidden[pix * 128 + c];
        else if (c < 254) hx[tid] = T(max(float(mconv[pix * 128 + c - 128]) + float(mbias[c - 128]), 0.0f));
        else hx[tid] = T(flow[pix * 2 + c - 254]);
    """,
)

# gates kernel: z from the first 128 channels of zr + czr; qx is (r from the last 128) * hidden, motion, flow
_GATES = metal_kernel(
    name="gru_gates",
    input_names=["zr", "czr", "hx"],
    output_names=["z", "qx"],
    source="""
        uint tid = thread_position_in_grid.x, c = tid % 128, pix = tid / 128, i = pix * 256 + c;
        z[tid] = T(1.0f / (1.0f + exp(-(float(zr[i]) + float(czr[i])))));
        float r = 1.0f / (1.0f + exp(-(float(zr[i + 128]) + float(czr[i + 128]))));
        qx[i] = T(r * float(hx[i]));
        qx[i + 128] = hx[i + 128];
    """,
)

# hidden' = (1 - z) hidden + z tanh(q + cq); FULL: hx' = [hidden' | motion | flow], else hidden' alone
_UPDATE = metal_kernel(
    name="gru_update",
    input_names=["q", "cq", "z", "hx"],
    output_names=["out"],
    source="""
        uint tid = thread_position_in_grid.x, c = tid % 128, pix = tid / 128, i = pix * 256 + c;
        float zz = float(z[tid]);
        T h = T((1.0f - zz) * float(hx[i]) + zz * precise::tanh(float(q[tid]) + float(cq[tid])));
        if (FULL) { out[i] = h; out[i + 128] = hx[i + 128]; } else { out[tid] = h; }
    """,
)


# ----------------------------------------------------------------------------- Winograd F(4, 5) for the GRU convs
# A 5-tap convolution over 4 neighbouring outputs needs 20 multiplications per channel pair; with the Toom-Cook
# points below it needs 8: transform 8 inputs (`wino_in`), one [8, tiles, C] @ [8, C, O] matmul, transform the 8
# products back (inside the gate kernels). All of it in float32, also for the float16 model (the transforms cancel
# digits, and the float32 matmul is as fast as the float16 one).
_POINTS = (0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5)


def _winograd(m=4, r=5):
    n = m + r - 1

    def vandermonde(cols):
        v = np.zeros((n, cols))
        for k, pt in enumerate(_POINTS):
            v[k] = [pt**i for i in range(cols)]
        v[n - 1, cols - 1] = 1.0  # the point at infinity
        return v

    return vandermonde(m).T, vandermonde(r), np.linalg.inv(vandermonde(n)).T  # A^T [m, n], G [n, r], B^T [n, n]


_AT, _G, _BT = _winograd()


def _msl(a):
    return "{" + ", ".join("{" + ", ".join(f"{v!r}f" for v in row) + "}" for row in a.astype(np.float32).tolist()) + "}"


_WINO = f"""
constant float WBT[8][8] = {_msl(_BT)};
constant float WAT[4][8] = {_msl(_AT)};
"""
# tile geometry shared by the three kernels: a tile is 4 consecutive positions along the convolution's axis
_TILE = """
    int n = VERT ? H : W, across = VERT ? W : (W + 3) / 4, stride = VERT ? W : 1;
    uint tiles = uint(VERT ? ((H + 3) / 4) * W : H * ((W + 3) / 4));
    int line = VERT ? int(tile % uint(W)) : int(tile / uint(across));
    int t0 = VERT ? int(tile / uint(W)) : int(tile % uint(across));
    int first = VERT ? line : line * W;  // pixel index of axis position a: first + a * stride
"""

_WINO_IN = metal_kernel(
    name="wino_in",
    input_names=["x"],
    output_names=["t"],
    header=_WINO,
    source="""
        int H = x_shape[1], W = x_shape[2];
        uint C = x_shape[3], lanes = C / 4;
        uint tid = thread_position_in_grid.x, c = (tid % lanes) * 4, tile = tid / lanes;
    """
    + _TILE
    + """
        float4 d[8];
        for (int j = 0; j < 8; ++j) {
            int a = 4 * t0 - 2 + j;
            uint i = uint(first + a * stride) * C + c;
            d[j] = (a >= 0 && a < n) ? float4(x[i], x[i + 1], x[i + 2], x[i + 3]) : float4(0.0f);
        }
        for (uint k = 0; k < 8; ++k) {
            float4 s = 0.0f;
            for (uint j = 0; j < 8; ++j) s += WBT[k][j] * d[j];
            uint o = (k * tiles + tile) * C + c;
            t[o] = s.x; t[o + 1] = s.y; t[o + 2] = s.z; t[o + 3] = s.w;
        }
    """,
)

# as gru_gates, from the Winograd products `m` [8, tiles, 256]; one thread per (tile, 4 channels)
_GATES_W = metal_kernel(
    name="gru_gates_wino",
    input_names=["m", "czr", "hx"],
    output_names=["z", "qx"],
    header=_WINO,
    source="""
        int H = hx_shape[1], W = hx_shape[2];
        uint tid = thread_position_in_grid.x, c = (tid % 32) * 4, tile = tid / 32;
    """
    + _TILE
    + """
        float4 vz[8], vr[8];
        for (uint k = 0; k < 8; ++k) {
            uint i = (k * tiles + tile) * 256 + c;
            vz[k] = float4(m[i], m[i + 1], m[i + 2], m[i + 3]);
            vr[k] = float4(m[i + 128], m[i + 129], m[i + 130], m[i + 131]);
        }
        for (int a = 0; a < 4 && 4 * t0 + a < n; ++a) {
            float4 sz = 0.0f, sr = 0.0f;
            for (uint k = 0; k < 8; ++k) { sz += WAT[a][k] * vz[k]; sr += WAT[a][k] * vr[k]; }
            uint pix = uint(first + (4 * t0 + a) * stride), i = pix * 256 + c, o = pix * 128 + c;
            sz += float4(czr[i], czr[i + 1], czr[i + 2], czr[i + 3]);
            sr += float4(czr[i + 128], czr[i + 129], czr[i + 130], czr[i + 131]);
            sz = 1.0f / (1.0f + exp(-sz));
            sr = 1.0f / (1.0f + exp(-sr)) * float4(hx[i], hx[i + 1], hx[i + 2], hx[i + 3]);
            for (uint e = 0; e < 4; ++e) { z[o + e] = T(sz[e]); qx[i + e] = T(sr[e]); qx[i + 128 + e] = hx[i + 128 + e]; }
        }
    """,
)

# as gru_update, from the Winograd products `m` [8, tiles, 128]
_UPDATE_W = metal_kernel(
    name="gru_update_wino",
    input_names=["m", "cq", "z", "hx"],
    output_names=["out"],
    header=_WINO,
    source="""
        int H = hx_shape[1], W = hx_shape[2];
        uint tid = thread_position_in_grid.x, c = (tid % 32) * 4, tile = tid / 32;
    """
    + _TILE
    + """
        float4 v[8];
        for (uint k = 0; k < 8; ++k) {
            uint i = (k * tiles + tile) * 128 + c;
            v[k] = float4(m[i], m[i + 1], m[i + 2], m[i + 3]);
        }
        for (int a = 0; a < 4 && 4 * t0 + a < n; ++a) {
            float4 s = 0.0f;
            for (uint k = 0; k < 8; ++k) s += WAT[a][k] * v[k];
            uint pix = uint(first + (4 * t0 + a) * stride), i = pix * 256 + c, o = pix * 128 + c;
            float4 zz = float4(z[o], z[o + 1], z[o + 2], z[o + 3]);
            s = precise::tanh(s + float4(cq[o], cq[o + 1], cq[o + 2], cq[o + 3]));
            s = (1.0f - zz) * float4(hx[i], hx[i + 1], hx[i + 2], hx[i + 3]) + zz * s;
            if (FULL) { for (uint e = 0; e < 4; ++e) { out[i + e] = T(s[e]); out[i + 128 + e] = hx[i + 128 + e]; } }
            else { for (uint e = 0; e < 4; ++e) out[o + e] = T(s[e]); }
        }
    """,
)


def extend(p32):
    """Extra float32 parameters of this variant: the GRU weights in the Winograd domain, `[8, in, out]`."""
    out = {}
    for half in (1, 2) if tuning().winograd else ():
        for gate in ("zr", "q"):
            w = np.array(p32[f"gru{half}.{gate}.weight"]).astype(np.float64)
            taps = w.reshape(w.shape[0], 5, w.shape[3])  # [out, tap, in]
            out[f"gru{half}.{gate}.wino"] = mx.array(np.einsum("kj,oji->kio", _G, taps).astype(np.float32))
    return out


def _wino_conv(x, u, vert):
    _, h, w, c = x.shape
    tiles = ((h + 3) // 4) * w if vert else h * ((w + 3) // 4)
    t = _WINO_IN(inputs=[x], template=[("VERT", vert)], grid=(tiles * (c // 4), 1, 1), threadgroup=_tg(),
                 output_shapes=[(8, tiles, c)], output_dtypes=[mx.float32])[0]  # fmt: skip
    return t @ u, tiles


def gru(p, hidden, motion, flow, czr1, cq1, czr2, cq2):
    """Both `SepConvGRU` halves; `motion` here is the raw 128-channel output of the motion encoder's last conv."""
    n, h, w, _ = hidden.shape
    t, px = hidden.dtype, h * w
    full, half = (n, h, w, 2 * HDIM), (n, h, w, HDIM)
    wino = "gru1.zr.wino" in p  # `extend` adds them when the tunable `winograd` is on
    hx = _PACK(inputs=[hidden, motion, p["update_block.encoder.conv.bias"], flow], template=[("T", t)],
               grid=(px * 256, 1, 1), threadgroup=_tg(), output_shapes=[full], output_dtypes=[t])[0]  # fmt: skip
    for i, (pad, czr, cq) in enumerate((((0, 2), czr1, cq1), ((2, 0), czr2, cq2)), 1):
        vert = i == 2
        if wino:
            m, tiles = _wino_conv(hx, p[f"gru{i}.zr.wino"], vert)
            z, qx = _GATES_W(inputs=[m, czr, hx], template=[("T", t), ("VERT", vert)], grid=(tiles * 32, 1, 1),
                             threadgroup=_tg(), output_shapes=[half, full], output_dtypes=[t, t])  # fmt: skip
            m, tiles = _wino_conv(qx, p[f"gru{i}.q.wino"], vert)
            hx = _UPDATE_W(inputs=[m, cq, z, hx], template=[("T", t), ("VERT", vert), ("FULL", i == 1)],
                           grid=(tiles * 32, 1, 1), threadgroup=_tg(), output_shapes=[full if i == 1 else half],
                           output_dtypes=[t])[0]  # fmt: skip
            continue
        zr = mx.conv2d(hx, p[f"gru{i}.zr.weight"], padding=pad)
        z, qx = _GATES(inputs=[zr, czr, hx], template=[("T", t)], grid=(px * 128, 1, 1), threadgroup=_tg(),
                       output_shapes=[half, full], output_dtypes=[t, t])  # fmt: skip
        q = mx.conv2d(qx, p[f"gru{i}.q.weight"], padding=pad)
        hx = _UPDATE(inputs=[q, cq, z, hx], template=[("T", t), ("FULL", i == 1)], grid=(px * 128, 1, 1),
                     threadgroup=_tg(), output_shapes=[full if i == 1 else half], output_dtypes=[t])[0]  # fmt: skip
    return hx


RAW_MOTION = True  # `gru` takes the motion conv before bias + relu (see model.motion_features)

# ----------------------------------------------------------------------------- instance norm
# per (row, channel) sum and sum of squares in float32; rows are then summed by MLX
_STATS = metal_kernel(
    name="inorm_stats",
    input_names=["x"],
    output_names=["s"],
    source="""
        uint W = x_shape[2], C = x_shape[3];
        uint tid = thread_position_in_grid.x, c = tid % C, row = tid / C;
        float a = 0.0f, b = 0.0f;
        uint i = row * W * C + c;
        for (uint k = 0; k < W; ++k, i += C) { float v = float(x[i]); a += v; b += v * v; }
        s[2 * tid] = a; s[2 * tid + 1] = b;
    """,
)

_APPLY = metal_kernel(
    name="inorm_apply",
    input_names=["x", "mean", "rstd", "add"],
    output_names=["y"],
    source="""
        uint H = x_shape[1], W = x_shape[2], C = x_shape[3];
        uint tid = thread_position_in_grid.x, c = tid % C, n = tid / (H * W * C);
        float v = (float(x[tid]) - mean[n * C + c]) * rstd[n * C + c];
        if (ADD) v += float(add[tid]);
        y[tid] = T(max(v, 0.0f));
    """,
)


def instance_norm_relu(x, add=None, eps=1e-5):
    n, h, w, c = x.shape
    s = _STATS(inputs=[x], template=[("T", x.dtype)], grid=(n * h * c, 1, 1), threadgroup=_tg(),
               output_shapes=[(n, h, c, 2)], output_dtypes=[mx.float32])[0].sum(axis=1) / (h * w)  # fmt: skip
    mean = s[..., 0]
    rstd = mx.rsqrt(mx.maximum(s[..., 1] - mean * mean, 0) + eps)
    return _APPLY(inputs=[x, mean, rstd, x if add is None else add], template=[("T", x.dtype), ("ADD", add is not None)],
                  grid=(x.size, 1, 1), threadgroup=_tg(), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]  # fmt: skip


# ----------------------------------------------------------------------------- convex upsampling
# one thread per output pixel: softmax over the 9 logits of its sub-position, weighted 3x3 flow sum (zero padded)
_CONVEX = metal_kernel(
    name="convex_up",
    input_names=["flow", "mask"],
    output_names=["out"],
    source="""
        int H = flow_shape[1], W = flow_shape[2];
        uint tid = thread_position_in_grid.x;
        int ox = int(tid % uint(4 * W)), oy = int(tid / uint(4 * W));
        int x = ox / 4, y = oy / 4;
        uint sub = uint((oy % 4) * 4 + (ox % 4)), m = uint(y * W + x) * 144 + sub;
        float l[9], top = -INFINITY;
        for (uint k = 0; k < 9; ++k) { l[k] = float(mask[m + 16 * k]); top = max(top, l[k]); }
        float sum = 0.0f, u = 0.0f, v = 0.0f;
        for (int k = 0; k < 9; ++k) {
            float e = exp(l[k] - top);
            sum += e;
            int nx = x + k % 3 - 1, ny = y + k / 3 - 1;
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                u += e * flow[2 * (ny * W + nx)]; v += e * flow[2 * (ny * W + nx) + 1];
            }
        }
        out[2 * tid] = 4.0f * u / sum; out[2 * tid + 1] = 4.0f * v / sum;
    """,
)


def convex_upsample(flow, mask):
    n, h, w, _ = flow.shape
    return _CONVEX(inputs=[flow, mask], grid=(16 * h * w, 1, 1), threadgroup=_tg(),
                   output_shapes=[(n, 4 * h, 4 * w, 2)], output_dtypes=[mx.float32])[0]  # fmt: skip
