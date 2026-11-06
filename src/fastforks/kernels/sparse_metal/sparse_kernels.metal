// Sparse 3-D convolution on Apple MPS (fastforks).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Submanifold 3x3x3 conv as one implicit gather GEMM:
//   out[n, o] = bias[o] + sum_v sum_c x[nbr[v, n], c] * w[v, c, o]
// x [N_in, C_in], nbr [27, N] (input row or -1), w [27, C_in, C_out], out [N, C_out]; C_in, C_out multiples of 32.
// One threadgroup (4 simdgroups) per 32 x 32 output tile, each simdgroup a 16 x 16 quarter as 2 x 2 8x8 matrices; K
// in steps of 32. A tap no voxel of the tile has is skipped. Accumulates in float32.
constant constexpr uint BM = 32, BN = 32, BK = 32, LD = 36;  // padded rows: fewer threadgroup bank conflicts

template <typename T>
kernel void subm_conv(device T* out [[buffer(0)]], device const T* x [[buffer(1)]], device const int* nbr [[buffer(2)]],
                      device const T* w [[buffer(3)]], device const T* bias [[buffer(4)]],
                      constant uint& N [[buffer(5)]], constant uint& C [[buffer(6)]], constant uint& O [[buffer(7)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    typedef vec<T, 4> T4;
    const uint tiles = O / BN;
    const uint m0 = (tg / tiles) * BM, n0 = (tg % tiles) * BN;
    threadgroup float A[BM * LD], B[BK * LD];
    threadgroup int rows[BM];
    threadgroup bool any;
    const uint r = tid / 4, c = (tid % 4) * 8;  // this thread's 8 elements of a 32 x 32 tile
    threadgroup float4* a4 = (threadgroup float4*)(A + r * LD + c);
    threadgroup float4* b4 = (threadgroup float4*)(B + r * LD + c);
    const uint sr = (sg / 2) * 16, sc = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint v = 0; v < 27; v++) {
        if (tid < BM) {  // simdgroup 0: this tap's input rows of the tile
            const uint n = m0 + tid;
            const int row = n < N ? nbr[v * N + n] : -1;
            rows[tid] = row;
            const bool hit = simd_any(row >= 0);
            if (tid == 0) any = hit;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!any) {  // uniform over the threadgroup
            threadgroup_barrier(mem_flags::mem_threadgroup);  // everyone has read `any` before the next tap writes it
            continue;
        }
        const int row = rows[r];
        device const T4* xs = (device const T4*)(x + uint(max(row, 0)) * C + c);
        device const T4* ws = (device const T4*)(w + (v * C + r) * O + n0 + c);
        for (uint k0 = 0; k0 < C; k0 += BK) {
            a4[0] = row >= 0 ? float4(xs[k0 / 4]) : 0.0f;
            a4[1] = row >= 0 ? float4(xs[k0 / 4 + 1]) : 0.0f;
            b4[0] = float4(ws[k0 * O / 4]);
            b4[1] = float4(ws[k0 * O / 4 + 1]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 a[2], b[2];
                for (uint i = 0; i < 2; i++) simdgroup_load(a[i], A + (sr + 8 * i) * LD + kk, LD);
                for (uint j = 0; j < 2; j++) simdgroup_load(b[j], B + kk * LD + sc + 8 * j, LD);
                for (uint i = 0; i < 2; i++)
                    for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], A + (sr + 8 * i) * LD + sc + 8 * j, LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = m0 + r;
    if (n < N) {
        device const T4* b = (device const T4*)(bias + n0 + c);
        device T4* o = (device T4*)(out + n * O + n0 + c);
        o[0] = T4(a4[0] + float4(b[0]));
        o[1] = T4(a4[1] + float4(b[1]));
    }
}

#define INSTANTIATE(T)                                                                                                  \
    template [[host_name("subm_conv_" #T)]] kernel void subm_conv<T>(device T*, device const T*, device const int*,    \
        device const T*, device const T*, constant uint&, constant uint&, constant uint&, uint, uint, uint);
INSTANTIATE(float)
INSTANTIATE(half)
