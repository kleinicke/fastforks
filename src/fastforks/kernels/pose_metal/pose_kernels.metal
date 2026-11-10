// Top-down pose on Apple MPS (fastforks): ViTPose's transposed-convolution decoder and DARK heatmap decoding,
// RTMPose's OpenCV-exact crop. Memory-bound glue around PyTorch's GEMMs and convolutions.
#include <metal_stdlib>
using namespace metal;

// NaN / (NaN or infinity) from the bits: holds whatever the compiler assumes about float comparisons (fast math).
static inline bool not_a_number(float a) { return (as_type<uint>(a) & 0x7fffffffu) > 0x7f800000u; }
static inline bool not_finite(float a) { return (as_type<uint>(a) & 0x7fffffffu) >= 0x7f800000u; }

// Overlap-add of a 4 x 4 stride-2 padding-1 transposed convolution computed as a GEMM, + bias, ReLU.
// cols [B, H * W, 16 * C] (input pixel, then tap ky * 4 + kx, then output channel), bias [C] ->
// out [B, 2H * 2W, C] (channels last). Output row oy takes input rows iy = (oy + 1) / 2 - {0, 1} with tap
// ky = oy + 1 - 2 iy (likewise for columns): four taps. One thread per output element, channels fastest.
template <typename T>
kernel void deconv4s2_relu(device T* out [[buffer(0)]], device const T* cols [[buffer(1)]],
                           device const T* bias [[buffer(2)]], constant uint& B [[buffer(3)]],
                           constant uint& H [[buffer(4)]], constant uint& W [[buffer(5)]],
                           constant uint& C [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint OW = 2 * W, OH = 2 * H;
    if (gid >= B * OH * OW * C) return;
    const uint c = gid % C, ox = (gid / C) % OW, oy = (gid / (C * OW)) % OH, b = gid / (C * OW * OH);
    float acc = float(bias[c]);
    const int ty = int(oy + 1) / 2, tx = int(ox + 1) / 2;
    for (int iy = ty - 1; iy <= ty; ++iy) {
        if (iy < 0 || iy >= int(H)) continue;
        const int ky = int(oy) + 1 - 2 * iy;
        for (int ix = tx - 1; ix <= tx; ++ix) {
            if (ix < 0 || ix >= int(W)) continue;
            const int kx = int(ox) + 1 - 2 * ix;
            acc += float(cols[((ulong(b) * H + iy) * W + ix) * 16 * C + (ky * 4 + kx) * C + c]);
        }
    }
    out[gid] = T(max(acc, 0.0f));
}


// DARK decoding (Zhang et al. 2020; `pose.models.vitpose.model.decode`) of M = P * K float32 heatmaps h x w: heatmap
// m = p * K + k starts at hm + p * h * w * K + k * cs, pixel i at i * ps (channels last [P, h * w, K]: cs = 1, ps = K;
// [P, K, h, w]: cs = h * w, ps = 1) -> out [M, 3]: x, y (heatmap pixels), score. The argmax (first maximum),
// then a Newton step on the log of the Gaussian-blurred heatmap (sigma, radius taps per side, half-sample reflect
// border, clipped to [lo, hi] before the log; a replicate border around the log map) from its 3 x 3 neighbourhood.
// One 256-thread group per heatmap; only the 9 blurred values at the neighbourhood are computed.
// NaN as in PyTorch: a heatmap with a NaN gives NaN x, y and score (`max` returns the NaN), a NaN blurred value
// (+inf next to -inf) NaN x, y; a constant map (all -inf too) decodes at pixel 0. Maps smaller than `radius` are
// rejected by the wrapper (PyTorch raises an index error there); the reflected taps are clamped all the same.
kernel void dark_decode_float(device float* out [[buffer(0)]], device const float* hm [[buffer(1)]],
                              constant uint& h [[buffer(2)]], constant uint& w [[buffer(3)]], constant uint& K [[buffer(4)]],
                              constant float& sigma [[buffer(5)]], constant int& radius [[buffer(6)]],
                              constant float& lo [[buffer(7)]], constant float& hi [[buffer(8)]],
                              constant uint& cs [[buffer(9)]], constant uint& ps [[buffer(10)]],
                              uint m [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_v[8];
    threadgroup uint best_i[8];
    threadgroup uint best_nan[8];  // [s]: SIMD group s saw a NaN; [0] after the merge: the heatmap has no maximum
    threadgroup float lg[9];
    const uint n = h * w;
    const float nan = as_type<float>(0x7fc00000u);
    if (n == 0) {  // nothing to read (all threads leave: no barrier is skipped by some)
        if (t < 3) out[m * 3 + t] = nan;
        return;
    }
    device const float* x = hm + ulong(m / K) * n * K + ulong(m % K) * cs;  // pixel i at x[i * ps]
    float v = -INFINITY;
    uint idx = 0xffffffffu;  // no element yet: any index is smaller, so a first -inf is taken too
    uint has_nan = 0;
    for (uint i = t; i < n; i += 256) {
        const float a = x[ulong(i) * ps];
        has_nan |= uint(not_a_number(a));
        if (a > v || (a == v && i < idx)) { v = a; idx = i; }  // strided scan: this thread's first maximum
    }
    for (uint off = 16; off > 0; off /= 2) {
        const float ov = simd_shuffle_down(v, off);
        const uint oi = simd_shuffle_down(idx, off);
        has_nan |= simd_shuffle_down(has_nan, off);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) { best_v[sg] = v; best_i[sg] = idx; best_nan[sg] = has_nan; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (uint s = 1; s < 8; ++s) {
            best_nan[0] |= best_nan[s];
            if (best_v[s] > best_v[0] || (best_v[s] == best_v[0] && best_i[s] < best_i[0])) {
                best_v[0] = best_v[s]; best_i[0] = best_i[s];
            }
        }
        if (best_nan[0] != 0 || best_i[0] >= n) {  // a NaN, or no element compared (all NaN): never index with it
            best_nan[0] = 1;
            best_i[0] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int xi = int(best_i[0] % w), yi = int(best_i[0] / w);
    if (t < 9) {  // blurred value at the clamped neighbour (dx, dy): the log map's replicate border
        const int px = clamp(xi + int(t % 3) - 1, 0, int(w) - 1), py = clamp(yi + int(t / 3) - 1, 0, int(h) - 1);
        float norm = 0;
        for (int k = -radius; k <= radius; ++k) norm += exp(-0.5f * (float(k) / sigma) * (float(k) / sigma));
        float acc = 0;
        for (int j = -radius; j <= radius; ++j) {
            int yy = py + j;
            yy = clamp(yy < 0 ? -yy - 1 : (yy >= int(h) ? 2 * int(h) - 1 - yy : yy), 0, int(h) - 1);
            float row = 0;
            for (int i = -radius; i <= radius; ++i) {
                int xx = px + i;
                xx = clamp(xx < 0 ? -xx - 1 : (xx >= int(w) ? 2 * int(w) - 1 - xx : xx), 0, int(w) - 1);
                row += exp(-0.5f * (float(i) / sigma) * (float(i) / sigma)) / norm * x[ulong(yy * int(w) + xx) * ps];
            }
            acc += exp(-0.5f * (float(j) / sigma) * (float(j) / sigma)) / norm * row;
        }
        lg[t] = not_a_number(acc) ? acc : log(clamp(acc, lo, hi));  // Metal's clamp would drop the NaN
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        if (best_nan[0] != 0) {
            out[m * 3] = nan; out[m * 3 + 1] = nan; out[m * 3 + 2] = nan;
            return;  // after the last barrier
        }
        // lg[(dy + 1) * 3 + (dx + 1)]
        const float i0 = lg[4];
        const float dx = 0.5f * (lg[5] - lg[3]), dy = 0.5f * (lg[7] - lg[1]);
        const float dxx = lg[5] - 2 * i0 + lg[3], dyy = lg[7] - 2 * i0 + lg[1];
        const float dxy = 0.5f * (lg[8] - lg[5] - lg[7] + 2 * i0 - lg[3] - lg[1] + lg[0]);
        const float eps = 1.1920928955078125e-07f;  // float32 machine epsilon
        const float a = dxx + eps, d = dyy + eps, det = a * d - dxy * dxy;
        out[m * 3] = float(xi) - (d * dx - dxy * dy) / det;
        out[m * 3 + 1] = float(yi) - (a * dy - dxy * dx) / det;
        out[m * 3 + 2] = best_v[0];
    }
}


// RTMPose's crops: OpenCV `warpAffine` (bilinear, zero border, fixed-point taps) of `image` [3, H, W] in [0, 1]
// to out [P, 3, h, w] in 0..255 integers, from the source coordinates x [P, w], y [P, h] (multiples of 1/32, from
// the runner): the image as uint8 levels (round(v * 255)), four taps with weights in 1/1024 (exact in float32, so
// the sum is exact in any order), rounded half up. One thread per output pixel, all three channels.
// A NaN or infinite coordinate (a box whose size overflows float32) gives NaN without reading the image, as the
// PyTorch crop (its tap weights are NaN there); finite coordinates outside the image give 0.
kernel void warp_crop_float(device float* out [[buffer(0)]], device const float* image [[buffer(1)]],
                            device const float* xs [[buffer(2)]], device const float* ys [[buffer(3)]],
                            constant uint& P [[buffer(4)]], constant uint& H [[buffer(5)]], constant uint& W [[buffer(6)]],
                            constant uint& h [[buffer(7)]], constant uint& w [[buffer(8)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= P * h * w) return;
    const uint j = gid % w, i = (gid / w) % h, p = gid / (w * h);
    const float x = xs[p * w + j], y = ys[p * h + i];
    if (not_finite(x) || not_finite(y)) {
        for (int c = 0; c < 3; ++c) out[((ulong(p) * 3 + c) * h + i) * w + j] = as_type<float>(0x7fc00000u);
        return;
    }
    const float x0 = floor(x), y0 = floor(y), fx = x - x0, fy = y - y0;
    float acc[3] = {0, 0, 0};
    for (int dy = 0; dy < 2; ++dy) {
        const float yy = y0 + dy, ty = dy ? fy : 1 - fy;
        if (!(yy >= 0 && yy < float(H))) continue;
        for (int dx = 0; dx < 2; ++dx) {
            const float xx = x0 + dx, tx = dx ? fx : 1 - fx;
            if (!(xx >= 0 && xx < float(W))) continue;
            const ulong o = ulong(yy) * W + ulong(xx);
            for (int c = 0; c < 3; ++c) acc[c] += rint(image[c * ulong(H) * W + o] * 255.0f) * (ty * tx);
        }
    }
    for (int c = 0; c < 3; ++c) out[((ulong(p) * 3 + c) * h + i) * w + j] = floor(acc[c] + 0.5f);
}

#define INSTANTIATE(T)                                                                                                  \
    template [[host_name("deconv4s2_relu_" #T)]] kernel void deconv4s2_relu<T>(device T*, device const T*,           \
        device const T*, constant uint&, constant uint&, constant uint&, constant uint&, uint);
INSTANTIATE(float)
INSTANTIATE(half)
