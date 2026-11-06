// VGGT's aggregator q / k LayerNorm + 2-D RoPE on Apple MPS (fastforks).
#include <metal_stdlib>
using namespace metal;

// qk [R, 2, H, 64] (the q | k half of the qkv projection, R = B * N token rows), cos / sin [R, 64] (per-token RoPE
// tables, sin with the rotation's sign folded in), w / b [2, 64] (q_norm, k_norm affine) -> out [2, B, H, N, 64]
// (q, k in scaled_dot_product_attention's layout):
//   a = LayerNorm(x) * w + b;  out = a * cos + swap(a) * sin,  swap: element j <-> j ^ 16 (the 16-wide quarters
//   inside each 32-wide half). One simdgroup per (row, q | k, head), two elements per thread; float32 statistics.
template <typename T>
kernel void qk_norm_rope(device T* out [[buffer(0)]], device const T* qk [[buffer(1)]], device const T* cs [[buffer(2)]],
                         device const T* sn [[buffer(3)]], device const T* w [[buffer(4)]], device const T* b [[buffer(5)]],
                         constant uint& R [[buffer(6)]], constant uint& N [[buffer(7)]], constant uint& H [[buffer(8)]],
                         constant float& eps [[buffer(9)]], uint gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint v = gid / 32, h = v % H, sel = (v / H) % 2, row = v / (2 * H);
    if (row >= R) return;
    device const T* x = qk + (ulong(row) * 2 + sel) * H * 64 + h * 64;
    const float x0 = x[lane], x1 = x[lane + 32];
    const float mean = simd_sum(x0 + x1) / 64;
    const float d0 = x0 - mean, d1 = x1 - mean;
    const float rstd = rsqrt(simd_sum(d0 * d0 + d1 * d1) / 64 + eps);
    const uint i0 = sel * 64 + lane, i1 = i0 + 32;
    const float a0 = d0 * rstd * float(w[i0]) + float(b[i0]);
    const float a1 = d1 * rstd * float(w[i1]) + float(b[i1]);
    const float p0 = simd_shuffle_xor(a0, 16), p1 = simd_shuffle_xor(a1, 16);
    device const T* c = cs + ulong(row) * 64;
    device const T* s = sn + ulong(row) * 64;
    const uint bi = row / N, ni = row % N;
    device T* o = out + ulong(sel) * R * H * 64 + ((ulong(bi) * H + h) * N + ni) * 64;
    o[lane] = T(a0 * float(c[lane]) + p0 * float(s[lane]));
    o[lane + 32] = T(a1 * float(c[lane + 32]) + p1 * float(s[lane + 32]));
}

#define INSTANTIATE(T)                                                                                                  \
    template [[host_name("qk_norm_rope_" #T)]] kernel void qk_norm_rope<T>(device T*, device const T*, device const T*, \
        device const T*, device const T*, device const T*, constant uint&, constant uint&, constant uint&,             \
        constant float&, uint, uint);
INSTANTIATE(float)
INSTANTIATE(half)
