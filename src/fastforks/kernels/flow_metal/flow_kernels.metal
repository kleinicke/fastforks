// Metal kernels for optical flow on Apple MPS (RAFT, SEA-RAFT). Each kernel is instantiated for float and half
// (`<name>_float`, `<name>_half`); the arithmetic is float32 in both.
#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIDE = 10;  // fixed: 2 r + 2 for the largest radius (4) the wrapper accepts
constant uint SIMDS = 4;      // fixed: SIMD groups per threadgroup (the wrapper's GROUP = 128 threads)
constant uint LANES = 8;      // fixed: lanes per window position, so a SIMD group takes 4 positions at a time

// Correlation window lookup without a correlation volume: level l of RAFT's pyramid is the correlation of f1 with f2
// average-pooled l times (pooling is linear), so each pixel's (2r + 2)^2 integer neighbours are dot products taken
// on the fly, then bilinearly interpolated to the (2r + 1)^2 window (= grid_sample, align_corners, zero padding).
//
// f1: [B, P, C] (pixels, channels last); f2: the levels one after another, level l as [B, h_l, w_l, C] starting at
// element meta[3l], h_l = meta[3l + 1], w_l = meta[3l + 2]; coords: [B, 2, P] float (x, y at level 0).
// out: [B, L * K * K, P], K = 2r + 1, channel l * K * K + i * K + j holds level l at (x / 2^l + i - r, y / 2^l + j - r).
// One SIMD group per (pixel, level): 4 window positions at a time, 8 lanes each splitting the C = 32 NV channels
// (NV a compile-time constant: f1 stays in registers, which more than doubles the speed over a runtime count).
template <typename T, uint NV>
kernel void corr_window(device T* out [[buffer(0)]], device const T* f1 [[buffer(1)]], device const T* f2 [[buffer(2)]],
                        device const float* coords [[buffer(3)]], device const uint* meta [[buffer(4)]],
                        constant uint& P [[buffer(5)]], constant uint& L [[buffer(6)]], constant int& r [[buffer(7)]],
                        constant float& scale [[buffer(8)]], uint gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint C = 32 * NV, G = 32 / LANES;
    threadgroup float dots[SIMDS][MAX_SIDE * MAX_SIDE + G];  // + G: the last pass may run past S * S
    const uint item = gid / 32;  // = (b * P + n) * L + l
    const uint l = item % L, bn = item / L, b = bn / P, n = bn - b * P;
    const uint S = uint(2 * r + 2), K = S - 1, sub = lane / LANES, s = lane % LANES;

    float4 a[NV];
    device const T* p1 = f1 + ulong(bn) * C + s * 4;
    for (uint v = 0; v < NV; ++v) a[v] = float4(*(device const vec<T, 4>*)(p1 + v * 4 * LANES));

    const float sc = 1.0f / float(1u << l);
    float cx = coords[ulong(b) * 2 * P + n] * sc, cy = coords[ulong(b) * 2 * P + P + n] * sc;
    const bool ok = isfinite(cx) && isfinite(cy) && fabs(cx) < 1.0e6f && fabs(cy) < 1.0e6f;
    if (!ok) cx = cy = -1.0e7f;  // every tap out of bounds: zeros
    const float x0 = floor(cx), y0 = floor(cy), ax = cx - x0, ay = cy - y0;
    const int hl = int(meta[3 * l + 1]), wl = int(meta[3 * l + 2]);
    const int ix = int(clamp(x0, -2.0e6f, 2.0e6f)) - r, iy = int(clamp(y0, -2.0e6f, 2.0e6f)) - r;
    device const T* base = f2 + meta[3 * l] + ulong(b) * hl * wl * C + s * 4;

    for (uint pos = sub; pos < S * S + sub; pos += G) {  // the same pass count on every lane (the shuffles below)
        const int x = ix + int(pos / S), y = iy + int(pos % S);
        const bool in = pos < S * S && x >= 0 && x < wl && y >= 0 && y < hl;
        device const T* p2 = base + (ulong(clamp(y, 0, hl - 1)) * wl + clamp(x, 0, wl - 1)) * C;  // read, then masked
        float t = 0.0f;
        for (uint v = 0; v < NV; ++v) t += dot(a[v], float4(*(device const vec<T, 4>*)(p2 + v * 4 * LANES)));
        for (uint m = LANES / 2; m > 0; m /= 2) t += simd_shuffle_xor(t, m);
        if (s == 0) dots[sg][pos] = in ? t * scale : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float w00 = (1.0f - ax) * (1.0f - ay), w10 = ax * (1.0f - ay), w01 = (1.0f - ax) * ay, w11 = ax * ay;
    device T* o = out + (ulong(b) * L + l) * K * K * P + n;
    for (uint k = lane; k < K * K; k += 32) {
        const uint i = k / K, j = k - i * K, d = i * S + j;
        const float v = w00 * dots[sg][d] + w10 * dots[sg][d + S] + w01 * dots[sg][d + 1] + w11 * dots[sg][d + S + 1];
        o[ulong(k) * P] = T(v);
    }
}

#define INSTANTIATE(T, C)                                                                                               \
    template [[host_name("corr_window_c" #C "_" #T)]] kernel void corr_window<T, C / 32>(                             \
        device T*, device const T*, device const T*, device const float*, device const uint*, constant uint&,          \
        constant uint&, constant int&, constant float&, uint, uint, uint);
INSTANTIATE(float, 128)
INSTANTIATE(float, 256)
INSTANTIATE(float, 384)
INSTANTIATE(float, 512)
INSTANTIATE(half, 128)
INSTANTIATE(half, 256)
INSTANTIATE(half, 384)
INSTANTIATE(half, 512)
