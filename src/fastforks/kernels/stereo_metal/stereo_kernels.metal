// Metal kernels for stereo models on Apple MPS (RAFT-Stereo, the IGEV family, CFNet, LaS2, HITNet, AANet, CREStereo,
// NMRF / BridgeDepth). Each kernel is instantiated for float and half (`<name>_float`, `<name>_half`); the arithmetic
// is float32 in both. One thread per output element (pixel for HITNet's, query token and head for the attention).
#include <metal_stdlib>
using namespace metal;

inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// ConvGRU gates. zr: [B, 2C, P] (the z conv's C channels, then the r conv's), cz / cr / h: [B, C, P], buf: [B, Cb, P]
// whose channels [0, C) hold h. Writes z = sigmoid(zr_z + cz) and, over h in buf, r * h with r = sigmoid(zr_r + cr).
template <typename T>
kernel void gru_gates(device T* z [[buffer(0)]], device T* buf [[buffer(1)]], constant T* zr [[buffer(2)]],
                      constant T* cz [[buffer(3)]], constant T* cr [[buffer(4)]], constant T* h [[buffer(5)]],
                      constant uint& C [[buffer(6)]], constant uint& P [[buffer(7)]], constant uint& Cb [[buffer(8)]],
                      uint i [[thread_position_in_grid]]) {
    uint cp = C * P, b = i / cp, j = i - b * cp;
    uint zi = b * 2 * cp + j;
    z[i] = T(sigmoid(float(zr[zi]) + float(cz[i])));
    buf[b * Cb * P + j] = T(sigmoid(float(zr[zi + cp]) + float(cr[i])) * float(h[i]));
}

// ConvGRU update: out = (1 - z) h + z tanh(q + cq).
template <typename T>
kernel void gru_update(device T* out [[buffer(0)]], constant T* h [[buffer(1)]], constant T* z [[buffer(2)]],
                       constant T* q [[buffer(3)]], constant T* cq [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    float zz = float(z[i]), hh = float(h[i]);
    out[i] = T(hh + zz * (precise::tanh(float(q[i]) + float(cq[i])) - hh));
}

// 1-D correlation lookup over all pyramid levels. vol: the levels one after another, level l as N = B*P rows of width
// meta[2l + 1] starting at meta[2l]; coords: [B, 2, P] float (x = channel 0). out: [B, L * (2r + 1), P], channel
// l * (2r + 1) + k holds level l at x / 2^l + k - r, linearly interpolated; taps outside the row read 0.
template <typename T>
kernel void corr_lookup(device T* out [[buffer(0)]], constant T* vol [[buffer(1)]], constant float* coords [[buffer(2)]],
                        constant uint* meta [[buffer(3)]], constant uint& P [[buffer(4)]],
                        constant uint& L [[buffer(5)]], constant int& r [[buffer(6)]],
                        uint i [[thread_position_in_grid]]) {
    uint K = 2 * r + 1, n = i % P, rest = i / P;  // i = (b * L * K + c) * P + n
    uint c = rest % (L * K), b = rest / (L * K);
    uint l = c / K;
    int t = int(c - l * K) - r;
    float x = coords[b * 2 * P + n] / float(1u << l);
    float x0 = floor(x), a = x - x0;
    int w = int(meta[2 * l + 1]);
    uint row = meta[2 * l] + (b * P + n) * uint(w);
    int i0 = int(x0) + t, i1 = i0 + 1;
    float v0 = (i0 >= 0 && i0 < w) ? float(vol[row + i0]) : 0.0f;
    float v1 = (i1 >= 0 && i1 < w) ? float(vol[row + i1]) : 0.0f;
    out[i] = T(v0 * (1.0f - a) + v1 * a);
}

// Geometry / correlation lookup of the IGEV family (`FastGeoEncodingVolume`), all segments of one output in one
// kernel. out: [N, Ct] (NHWC rows, N = B*H*W pixels); disp: [N] float. seg: S segments of 6 uint each, by ascending
// first channel: first output channel, source (0: geo, 1: corr), offset into that source, C, width D, scale l. A geo
// segment reads rows [N, C, D] of `geo` at x = disp / 2^l + k - r (channel c * K + k); a corr segment reads rows
// [N, D] of `corr` at x = (n % W) / 2^l - disp / 2^l + k - r (channel k, C = 1). Linear interpolation, taps outside
// the row read 0 (= grid_sample, align_corners, zero padding). G: the geometry volume's type.
template <typename T, typename G>
kernel void line_lookup(device T* out [[buffer(0)]], constant G* geo [[buffer(1)]], constant T* corr [[buffer(2)]],
                        constant float* disp [[buffer(3)]], constant uint* seg [[buffer(4)]],
                        constant uint& S [[buffer(5)]], constant uint& Ct [[buffer(6)]], constant uint& W [[buffer(7)]],
                        constant int& r [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    uint n = i / Ct, ch = i - n * Ct, s = 0;
    while (s + 1 < S && seg[6 * (s + 1)] <= ch) s++;
    constant uint* m = seg + 6 * s;
    uint K = 2 * r + 1, c = ch - m[0], C = m[3];
    int D = int(m[4]);
    float sc = 1.0f / float(1u << m[5]), d = disp[n];
    uint g = c / K;
    int k = int(c - g * K) - r;
    float x = m[1] == 0 ? d * sc + float(k) : float(n % W) * sc - d * sc + float(k);
    float x0 = floor(x), a = x - x0;
    int i0 = int(x0), i1 = i0 + 1;
    uint row = m[2] + (n * C + g) * uint(D);
    float v0 = 0.0f, v1 = 0.0f;
    if (m[1] == 0) {
        if (i0 >= 0 && i0 < D) v0 = float(geo[row + i0]) * (1.0f - a);
        if (i1 >= 0 && i1 < D) v1 = float(geo[row + i1]) * a;
    } else {
        if (i0 >= 0 && i0 < D) v0 = float(corr[row + i0]) * (1.0f - a);
        if (i1 >= 0 && i1 < D) v1 = float(corr[row + i1]) * a;
    }
    out[i] = T(v0 + v1);
}

// CFNet's refinement-stage volume [B, G + 2 Cc + 1, D, h, W] for rows [a, a + h) of the features (H rows): per
// disparity sample ds [B, D, h, W] the right features are read at column trunc(x - ds) (zero outside [0, W - 1]):
// channels [0, G) the group-wise correlation of gl / gr (Cg channels, mean over Cg / G), then the left concat
// features cl, the warped right ones cr (Cc channels each), then ds itself.
template <typename T>
kernel void warp_volume(device T* out [[buffer(0)]], constant T* gl [[buffer(1)]], constant T* gr [[buffer(2)]],
                        constant T* cl [[buffer(3)]], constant T* cr [[buffer(4)]], constant float* ds [[buffer(5)]],
                        constant uint& G [[buffer(6)]], constant uint& Cg [[buffer(7)]], constant uint& Cc [[buffer(8)]],
                        constant uint& D [[buffer(9)]], constant uint& H [[buffer(10)]], constant uint& W [[buffer(11)]],
                        constant uint& a [[buffer(12)]], constant uint& h [[buffer(13)]],
                        uint i [[thread_position_in_grid]]) {
    uint Ct = G + 2 * Cc + 1, x = i % W, t = i / W, y = t % h;
    t /= h;
    uint d = t % D;
    t /= D;
    uint ch = t % Ct, b = t / Ct;
    float s = ds[((b * D + d) * h + y) * W + x];
    if (ch == Ct - 1) { out[i] = T(s); return; }
    uint row = (a + y) * W;
    if (ch >= G && ch < G + Cc) { out[i] = cl[(b * Cc + ch - G) * H * W + row + x]; return; }
    float xr = float(x) - s;
    if (xr < 0.0f || xr > float(W - 1)) { out[i] = T(0.0f); return; }
    uint col = row + uint(xr);
    if (ch >= G) { out[i] = cr[(b * Cc + ch - G - Cc) * H * W + col]; return; }
    uint n = Cg / G;
    float acc = 0.0f;
    for (uint c = ch * n; c < ch * n + n; c++) {
        uint o = (b * Cg + c) * H * W;
        acc += float(gl[o + row + x]) * float(gr[o + col]);
    }
    out[i] = T(acc / float(n));
}

// Group-wise correlation volume [B, G, D, H, W] of l / r [B, C, H, W]: the mean over each group's C / G channels of
// l(x) * r(x - d), 0 for x < d (the plain correlation volume: G = 1).
template <typename T>
kernel void gwc_volume(device T* out [[buffer(0)]], constant T* l [[buffer(1)]], constant T* r [[buffer(2)]],
                       constant uint& G [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& D [[buffer(5)]],
                       constant uint& H [[buffer(6)]], constant uint& W [[buffer(7)]],
                       uint i [[thread_position_in_grid]]) {
    uint x = i % W, t = i / W, y = t % H;
    t /= H;
    uint d = t % D;
    t /= D;
    uint g = t % G, b = t / G;
    if (x < d) { out[i] = T(0.0f); return; }
    uint n = C / G, P = H * W, o = (b * C + g * n) * P + y * W + x;
    float acc = 0.0f;
    for (uint c = 0; c < n; c++, o += P) acc += float(l[o]) * float(r[o - d]);
    out[i] = T(acc / float(n));
}

// HITNet level-init cost of tile (n, y, x): lt [N, C, h, w], rt [N, C, h, Wr] (every pixel column),
// cost(d) = sum_c |lt(x) - rt(max(0, 4x + 1 - d))| for d < D, each difference rounded to T (as PyTorch in T). Writes
// the first minimizing d and its cost (float [N, h, w]).
template <typename T>
kernel void hit_init_cost(device float* disp [[buffer(0)]], device float* cost [[buffer(1)]],
                          constant T* lt [[buffer(2)]], constant T* rt [[buffer(3)]], constant uint& C [[buffer(4)]],
                          constant uint& h [[buffer(5)]], constant uint& w [[buffer(6)]],
                          constant uint& Wr [[buffer(7)]], constant uint& D [[buffer(8)]],
                          uint i [[thread_position_in_grid]]) {
    uint x = i % w, t = i / w, y = t % h, n = t / h;
    float best = FLT_MAX;  // not INFINITY: Metal compiles with fast math
    uint bd = 0;
    for (uint d = 0; d < D; d++) {
        uint col = uint(max(0, int(4 * x + 1) - int(d)));
        float s = 0.0f;
        for (uint c = 0; c < C; c++) {
            uint row = (n * C + c) * h + y;
            s += abs(float(T(float(lt[row * w + x]) - float(rt[row * Wr + col]))));
        }
        if (s < best) { best = s; bd = d; }
    }
    disp[i] = float(bd);
    cost[i] = best;
}

// sum_c |fl(Y, X) - fr(Y, x)| with fr read at column x by clamped linear interpolation (grid_sample, border padding).
template <typename T>
inline float warp_sad(constant T* fl, constant T* fr, uint n, uint C, uint H, uint W, uint Wr, uint Y, uint X,
                      float x) {
    x = clamp(x, 0.0f, float(Wr - 1));
    float x0 = floor(x), a = x - x0;
    uint i0 = min(uint(x0), Wr - 1), i1 = min(i0 + 1, Wr - 1);  // in bounds for a NaN x too
    float s = 0.0f;
    for (uint c = 0; c < C; c++) {
        constant T* r = fr + ((n * C + c) * H + Y) * Wr;
        s += abs(float(fl[((n * C + c) * H + Y) * W + X]) - (float(r[i0]) * (1.0f - a) + float(r[i1]) * a));
    }
    return s;
}

// HITNet tile features of hypotheses hyp [N, 16, h, w] (d, dx, dy, ...) on fl / fr [N, C, 4h, W | Wr]: per pixel
// (Y, X) = (4y + by, 4x + bx) at plane disparity e = d + (bx - 1.5) dx + (by - 1.5) dy, [sum_c |fl|, SAD at X - e + 1,
// X - e, X - e - 1]; out: float [N, 64, h, w], channel (4 by + bx) * 4 + k.
template <typename T>
kernel void hit_tile_sad(device float* out [[buffer(0)]], constant float* hyp [[buffer(1)]],
                         constant T* fl [[buffer(2)]], constant T* fr [[buffer(3)]], constant uint& C [[buffer(4)]],
                         constant uint& h [[buffer(5)]], constant uint& w [[buffer(6)]], constant uint& Wr [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
    uint W = 4 * w, H = 4 * h, X = i % W, t = i / W, Y = t % H, n = t / H;
    uint x = X / 4, y = Y / 4, bx = X % 4, by = Y % 4, P = h * w, p = y * w + x;
    constant float* hp = hyp + n * 16 * P + p;
    float e = hp[0] + (float(bx) - 1.5f) * hp[P] + (float(by) - 1.5f) * hp[2 * P];
    float base = float(X) - e, a = 0.0f;
    for (uint c = 0; c < C; c++) a += abs(float(fl[((n * C + c) * H + Y) * W + X]));
    device float* o = out + (n * 64 + (by * 4 + bx) * 4) * P + p;
    o[0] = a;
    o[P] = warp_sad(fl, fr, n, C, H, W, Wr, Y, X, base + 1.0f);
    o[2 * P] = warp_sad(fl, fr, n, C, H, W, Wr, Y, X, base);
    o[3 * P] = warp_sad(fl, fr, n, C, H, W, Wr, Y, X, base - 1.0f);
}

// HITNet refinement SAD at disparity d [N, 1, H, W] on fl / fr [N, C, H, W | Wr]: out float [N, 3, H, W], the SADs
// at X - d, X - (d + 1), X - (d - 1).
template <typename T>
kernel void hit_refine_sad(device float* out [[buffer(0)]], constant float* disp [[buffer(1)]],
                           constant T* fl [[buffer(2)]], constant T* fr [[buffer(3)]], constant uint& C [[buffer(4)]],
                           constant uint& H [[buffer(5)]], constant uint& W [[buffer(6)]],
                           constant uint& Wr [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    uint P = H * W, X = i % W, t = i / W, Y = t % H, n = t / H;
    float d = disp[i];
    device float* o = out + n * 3 * P + Y * W + X;
    o[0] = warp_sad(fl, fr, n, C, H, W, Wr, Y, X, float(X) - d);
    o[P] = warp_sad(fl, fr, n, C, H, W, Wr, Y, X, float(X) - (d + 1.0f));
    o[2 * P] = warp_sad(fl, fr, n, C, H, W, Wr, Y, X, float(X) - (d - 1.0f));
}

// deform_cols: AANet's (modulated) deformable-convolution columns, cols[(c * K + k) * n * Wo + y * Wo + x] for output
// rows r0 .. r0 + n of one image: the input sampled bilinearly (zeros outside) at the tap shifted by its offset, times
// the mask. g = [C, H, W, Ho, Wo, kw, K, dg, stride y/x, padding y/x, dilation y/x, r0, n, has mask]
template <typename T>
kernel void deform_cols(device T* cols [[buffer(0)]], constant T* src [[buffer(1)]], constant float* off [[buffer(2)]],
                        constant T* mask [[buffer(3)]], constant int* g [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    int C = g[0], H = g[1], W = g[2], Ho = g[3], Wo = g[4], kw = g[5], K = g[6], dg = g[7], n = g[15];
    int xo = int(i) % Wo, t = int(i) / Wo, yl = t % n;
    t /= n;
    int k = t % K, c = t / K, grp = c / (C / dg), yo = g[14] + yl, P = Ho * Wo;
    int at = yo * Wo + xo, ok = (grp * K + k) * 2 * P + at;
    float py = float(yo * g[8] - g[10] + k / kw * g[12]) + off[ok];
    float px = float(xo * g[9] - g[11] + k % kw * g[13]) + off[ok + P];
    float y0 = floor(py), x0 = floor(px), wy = py - y0, wx = px - x0, v = 0.0f;
    int iy = int(y0), ix = int(x0);
    constant T* s = src + c * H * W;
    for (int a = 0; a < 2; a++) {
        int yy = iy + a;
        if (yy < 0 || yy >= H) continue;
        float fy = a ? wy : 1.0f - wy;
        for (int e = 0; e < 2; e++) {
            int xx = ix + e;
            if (xx >= 0 && xx < W) v += fy * (e ? wx : 1.0f - wx) * float(s[yy * W + xx]);
        }
    }
    if (g[16]) v *= float(mask[(grp * K + k) * P + at]);
    cols[i] = T(v);
}

// CREStereo's AGCL `corr_iter`, step 1: f2 [N, C, H, W] (float) warped by the flow (float [N, 2, H, W]; bilinear,
// zero outside) into out [N, C, H, W]; one thread per pixel, all channels.
template <typename T>
kernel void agcl_warp(device T* out [[buffer(0)]], constant float* f2 [[buffer(1)]], constant float* flow [[buffer(2)]],
                      constant uint& C [[buffer(3)]], constant uint& H [[buffer(4)]], constant uint& W [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
    uint HW = H * W, p = i % HW, n = i / HW;
    float px = float(p % W) + flow[n * 2 * HW + p], py = float(p / W) + flow[(n * 2 + 1) * HW + p];
    float x0 = floor(px), y0 = floor(py), ax = px - x0, ay = py - y0;
    int at[4];
    float wt[4];
    for (int j = 0; j < 4; j++) {
        int sy = int(y0) + j / 2, sx = int(x0) + j % 2;
        bool in = sy >= 0 && sy < int(H) && sx >= 0 && sx < int(W);
        at[j] = in ? sy * int(W) + sx : 0;
        wt[j] = in ? (j / 2 ? ay : 1.0f - ay) * (j % 2 ? ax : 1.0f - ax) : 0.0f;
    }
    for (uint c = 0; c < C; c++) {
        constant float* s = f2 + (n * C + c) * HW;
        out[(n * C + c) * HW + p] = T(wt[0] * s[at[0]] + wt[1] * s[at[1]] + wt[2] * s[at[2]] + wt[3] * s[at[3]]);
    }
}

// Step 2: the mean over each group's C / G channels of f1 * warped at the 9 positions of a (2 ry + 1) x (2 rx + 1)
// window around the pixel, the window clamped to the image (replicate padding). out [N, G * 9, H, W] (channel
// g * 9 + window index), f1 / warped [N, C, H, W]; one thread per output.
template <typename T>
kernel void agcl_window(device T* out [[buffer(0)]], constant T* f1 [[buffer(1)]], constant T* warped [[buffer(2)]],
                        constant uint& G [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& H [[buffer(5)]],
                        constant uint& W [[buffer(6)]], constant int& ry [[buffer(7)]], constant int& rx [[buffer(8)]],
                        uint i [[thread_position_in_grid]]) {
    uint HW = H * W, x = i % W, t = i / W, y = t % H;
    t /= H;
    int k = t % 9;
    t /= 9;
    uint g = t % G, n = t / G;
    int yy = clamp(int(y) + k / (2 * rx + 1) - ry, 0, int(H) - 1);
    int xx = clamp(int(x) + k % (2 * rx + 1) - rx, 0, int(W) - 1);
    uint m = C / G;
    float acc = 0.0f;
    for (uint c = g * m; c < g * m + m; c++) {
        uint o = (n * C + c) * HW;
        acc += float(f1[o + y * W + x]) * float(warped[o + yy * W + xx]);
    }
    out[i] = T(acc / float(m));
}

// 1-D correlation lookup around a disparity (DEFOM-Stereo's `CorrBlock1D`). vol, meta: as corr_lookup (row width
// meta[2l + 1] = W at level 0); disp: [B, P] float; seg: S (level, scale) pairs. out: [B, S * (2r + 1), P], channel
// s * (2r + 1) + k holds level l_s at (x - scale_s * disp) / 2^l_s + k - r (x = n % W: the pixel's column), linearly
// interpolated; taps outside the row read 0.
template <typename T>
kernel void disp_lookup(device T* out [[buffer(0)]], constant T* vol [[buffer(1)]], constant float* disp [[buffer(2)]],
                        constant uint* meta [[buffer(3)]], constant float* seg [[buffer(4)]],
                        constant uint& P [[buffer(5)]], constant uint& S [[buffer(6)]], constant int& r [[buffer(7)]],
                        uint i [[thread_position_in_grid]]) {
    uint K = 2 * r + 1, n = i % P, rest = i / P;  // i = (b * S * K + c) * P + n
    uint c = rest % (S * K), b = rest / (S * K);
    uint s = c / K;
    int t = int(c - s * K) - r;
    uint l = uint(seg[2 * s]);
    float x = (float(n % meta[1]) - seg[2 * s + 1] * disp[b * P + n]) / float(1u << l);
    float x0 = floor(x), a = x - x0;
    int w = int(meta[2 * l + 1]);
    uint row = meta[2 * l] + (b * P + n) * uint(w);
    int i0 = int(x0) + t, i1 = i0 + 1;
    float v0 = (i0 >= 0 && i0 < w) ? float(vol[row + i0]) : 0.0f;
    float v1 = (i1 >= 0 && i1 < w) ? float(vol[row + i1]) : 0.0f;
    out[i] = T(v0 * (1.0f - a) + v1 * a);
}

// NMRF / BridgeDepth WindowAttention (Swin windows of ws x ws cells x n candidates = T tokens, head_dim 32) with
// relative-position embeddings on q, k and v (the CUDA `fastbb` win_rpe_attn's math):
//   logit_ij = q_i . k_j + q_i . Ek[r] + k_j . Eq[r] + mask[w % nW][i][j]      (q, Eq scaled by `scale`)
//   out_i    = sum_j softmax_j(logit_ij) * (v_j + Ev[r]),   r = (hi - hj + ws - 1) * (2 ws - 1) + (wi - wj + ws - 1)
// straight on the packed projection qkv [B, Hc, Wc, n, 3C] (q | k | v, head h at column h * 32 of each), windows taken
// after a cyclic shift by -shift; out [B, Hc, Wc, n, C] in place of the tokens. table: [(2 ws - 1)^2, 3C] (per head
// Eq | Ek | Ev). One thread per (window, head, query token), online softmax. meta: heads, T, ws, n, Hc, Wc, shift, nW.
template <typename T>
kernel void win_rpe_attn(device T* out [[buffer(0)]], device const T* qkv [[buffer(1)]], device const T* table [[buffer(2)]],
                         device const float* mask [[buffer(3)]], constant uint* meta [[buffer(4)]],
                         constant float& scale [[buffer(5)]], uint idx [[thread_position_in_grid]]) {
    const uint heads = meta[0], Tn = meta[1], ws = meta[2], n = meta[3], Hc = meta[4], Wc = meta[5], shift = meta[6];
    const uint nW = meta[7], C = heads * 32, C3 = 3 * C, nwy = Hc / ws, nwx = Wc / ws;
    const uint i = idx % Tn, wh = idx / Tn, h = wh % heads, w = wh / heads;
    const uint b = w / (nwy * nwx), wr = w % (nwy * nwx), wy = wr / nwx, wx = wr % nwx;
    auto row_of = [&](uint t) -> uint {
        uint cell = t / n, cy = cell / ws, cx = cell % ws;
        uint y = wy * ws + cy + shift, x = wx * ws + cx + shift;
        if (y >= Hc) y -= Hc;
        if (x >= Wc) x -= Wc;
        return ((b * Hc + y) * Wc + x) * n + t % n;
    };
    const uint row = row_of(i);
    typedef vec<T, 4> T4;  // 8 x 4 channels, read as vectors
    float4 q[8], acc[8];
    device const T4* q4 = (device const T4*)(qkv + row * C3 + h * 32);
    for (uint e = 0; e < 8; e++) {
        q[e] = float4(q4[e]) * scale;
        acc[e] = 0.0f;
    }
    const uint ci = i / n, hi = ci / ws, wi = ci % ws;
    device const float* mrow = mask + ((w % nW) * Tn + i) * Tn;
    float m = -FLT_MAX, s = 0.0f;  // finite sentinels, not INFINITY: Metal compiles with fast math (no infinities)
    for (uint j = 0; j < Tn; j++) {
        float l = mrow[j];
        if (l < -1e30f) continue;  // a masked pair (-inf or a large negative mask value)
        const uint cj = j / n, hj = cj / ws, wj = cj % ws;
        const uint r = (hi + ws - 1 - hj) * (2 * ws - 1) + (wi + ws - 1 - wj);
        device const T4* eq = (device const T4*)(table + r * C3 + h * 96);
        device const T4* kj = (device const T4*)(qkv + row_of(j) * C3 + C + h * 32);
        float a = 0.0f, bq = 0.0f;
        for (uint e = 0; e < 8; e++) {
            float4 k = float4(kj[e]);
            a += dot(q[e], k + float4(eq[8 + e]));
            bq += dot(k, float4(eq[e]));
        }
        l += a + bq * scale;
        float mn = max(m, l), c = exp(m - mn), p = exp(l - mn);
        s = s * c + p;
        device const T4* vj = kj + C / 4;
        for (uint e = 0; e < 8; e++) acc[e] = acc[e] * c + p * (float4(vj[e]) + float4(eq[16 + e]));
        m = mn;
    }
    device vec<T, 4>* o = (device vec<T, 4>*)(out + row * C + h * 32);
    for (uint e = 0; e < 8; e++) o[e] = T4(acc[e] / s);
}

#define INSTANTIATE(T)                                                                                                  \
    template [[host_name("win_rpe_attn_" #T)]] kernel void win_rpe_attn<T>(device T*, device const T*,                \
        device const T*, device const float*, constant uint*, constant float&, uint);                                                        \
    template [[host_name("gru_gates_" #T)]] kernel void gru_gates<T>(                                                 \
        device T*, device T*, constant T*, constant T*, constant T*, constant T*, constant uint&, constant uint&,       \
        constant uint&, uint);                                                                                          \
    template [[host_name("gru_update_" #T)]] kernel void gru_update<T>(device T*, constant T*, constant T*,           \
                                                                       constant T*, constant T*, uint);                \
    template [[host_name("corr_lookup_" #T)]] kernel void corr_lookup<T>(                                             \
        device T*, constant T*, constant float*, constant uint*, constant uint&, constant uint&, constant int&, uint);  \
    template [[host_name("warp_volume_" #T)]] kernel void warp_volume<T>(                                             \
        device T*, constant T*, constant T*, constant T*, constant T*, constant float*, constant uint&, constant uint&, \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint);         \
    template [[host_name("gwc_volume_" #T)]] kernel void gwc_volume<T>(device T*, constant T*, constant T*,           \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint);                          \
    template [[host_name("hit_init_cost_" #T)]] kernel void hit_init_cost<T>(device float*, device float*,            \
        constant T*, constant T*, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,      \
        uint);                                                                                                          \
    template [[host_name("hit_tile_sad_" #T)]] kernel void hit_tile_sad<T>(device float*, constant float*,            \
        constant T*, constant T*, constant uint&, constant uint&, constant uint&, constant uint&, uint);               \
    template [[host_name("hit_refine_sad_" #T)]] kernel void hit_refine_sad<T>(device float*, constant float*,        \
        constant T*, constant T*, constant uint&, constant uint&, constant uint&, constant uint&, uint);               \
    template [[host_name("deform_cols_" #T)]] kernel void deform_cols<T>(device T*, constant T*, constant float*,       \
        constant T*, constant int*, uint);                                                                              \
    template [[host_name("agcl_warp_" #T)]] kernel void agcl_warp<T>(device T*, constant float*, constant float*,     \
        constant uint&, constant uint&, constant uint&, uint);                                                          \
    template [[host_name("agcl_window_" #T)]] kernel void agcl_window<T>(device T*, constant T*, constant T*,         \
        constant uint&, constant uint&, constant uint&, constant uint&, constant int&, constant int&, uint);             \
    template [[host_name("disp_lookup_" #T)]] kernel void disp_lookup<T>(device T*, constant T*, constant float*,     \
        constant uint*, constant float*, constant uint&, constant uint&, constant int&, uint);
INSTANTIATE(float)
INSTANTIATE(half)

#define INSTANTIATE_LOOKUP(T, G, NAME)                                                                                  \
    template [[host_name(NAME #T)]] kernel void line_lookup<T, G>(device T*, constant G*, constant T*, constant float*, \
        constant uint*, constant uint&, constant uint&, constant uint&, constant int&, uint);
INSTANTIATE_LOOKUP(float, float, "line_lookup_")
INSTANTIATE_LOOKUP(half, float, "line_lookup_")
INSTANTIATE_LOOKUP(half, half, "line_lookup_hgeo_")  // a float16 geometry volume (FoundationStereo: the memory)
