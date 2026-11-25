// MVSplat plane-sweep correlation: one SIMD group per (view, depth, pixel). Each lane takes 4-channel slices of the
// four bilinear taps of the other view and of the view's own features (NHWC, coalesced rows), fp32 products, one
// simd_sum. No [V, C, D, h, w] warped volume. The first offset writes the volume, later offsets add to it.
#include <metal_stdlib>
using namespace metal;

#define PLANE_SWEEP(T, NAME)                                                                                          \
kernel void NAME(device const T* f, device const float* grid, device float* out, constant uint& V, constant uint& D,  \
                 constant uint& H, constant uint& W, constant uint& C, constant uint& shift,                          \
                 uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {                     \
    uint index = gid / 32, hw = H * W;                                                                                \
    if (index >= V * D * hw) return;                                                                                  \
    uint pix = index % hw, n = index / (D * hw), m = (n + shift) % V;                                                 \
    float ix = (grid[2 * size_t(index)] + 1.0f) * float(W - 1) * 0.5f;                                                \
    float iy = (grid[2 * size_t(index) + 1] + 1.0f) * float(H - 1) * 0.5f;                                            \
    float fx = floor(ix), fy = floor(iy), ax = ix - fx, ay = iy - fy;                                                 \
    int x0 = int(fx), y0 = int(fy);                                                                                   \
    const device vec<T, 4>* o4 = (const device vec<T, 4>*)f + size_t(m) * hw * (C / 4);                               \
    const device vec<T, 4>* f4 = (const device vec<T, 4>*)f + (size_t(n) * hw + pix) * (C / 4);                       \
    float acc = 0.0f;                                                                                                 \
    for (uint cv = lane; cv < C / 4; cv += 32) {                                                                      \
        float4 s = float4(0.0f);                                                                                      \
        for (int dy = 0; dy < 2; ++dy) {                                                                              \
            int yy = y0 + dy;                                                                                         \
            if (yy < 0 || yy >= int(H)) continue;                                                                     \
            float wy = dy ? ay : 1.0f - ay;                                                                           \
            for (int dx = 0; dx < 2; ++dx) {                                                                          \
                int xx = x0 + dx;                                                                                     \
                if (xx < 0 || xx >= int(W)) continue;                                                                 \
                s += float4(o4[(size_t(yy) * W + xx) * (C / 4) + cv]) * (wy * (dx ? ax : 1.0f - ax));                 \
            }                                                                                                         \
        }                                                                                                             \
        acc += dot(float4(f4[cv]), s);                                                                                \
    }                                                                                                                 \
    acc = simd_sum(acc);                                                                                              \
    if (lane == 0) out[index] = shift == 1 ? acc : out[index] + acc;                                                  \
}

PLANE_SWEEP(float, plane_sweep_float)
PLANE_SWEEP(half, plane_sweep_half)
