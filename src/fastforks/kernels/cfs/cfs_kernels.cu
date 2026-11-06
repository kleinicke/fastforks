// C-FoundationStereo-S disparity transformer (TAO's CostVolumeDisparityAttention, 'standard' q/k/v layout):
// d_model 28, 4 heads x 7 (padded to 8 for flash SDPA), dim_feedforward 28, post-norm layers. Two kernels per layer
// around torch's flash SDPA: tao_qkv (q|k|v projections into the padded head layout) and tao_post (out_proj + residual
// + LN1 + FFN + residual + LN2). fp16 storage, fp32 math, fp16 rounding where torch's fp16 path rounds.
#include <cuda_fp16.h>

#define TC 28

// TAO views q/k/v (N, L, 28) as (N, 4, L, 7) without a transpose: within one sequence the flat index f = l * 28 + c
// goes to head f / (7 L), position (f % (7 L)) / 7, dim f % 7; heads are padded to 8 dims (zeros).
__device__ __forceinline__ long long tao_slot(long long n, int L, int f) {
    int h = f / (7 * L), r = f - h * 7 * L;
    return ((n * 4 + h) * (long long)L + r / 7) * 8 + (r % 7);
}

// One warp per token, lane c = channel c (28 of 32 lanes); x[c] broadcast by shuffles, weights in shared memory.
#define TAO_WARPS(t, T) for (long long t = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 32; t < (T); \
                             t += ((long long)gridDim.x * blockDim.x) / 32)

__device__ __forceinline__ float tao_mv(const float* W, float x, int lane) {   // (W x)[lane], W row-major 28x28
    float acc = 0.f;
    const int j = lane < TC ? lane : 0;
    #pragma unroll
    for (int c = 0; c < TC; c++) acc += W[j * TC + c] * __shfl_sync(0xffffffffu, x, c);
    return acc;
}

__device__ __forceinline__ float tao_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float tao_ln(float v, bool on, float g, float b) {   // LayerNorm over the 28 lanes, eps 1e-5
    float m = tao_warp_sum(on ? v : 0.f) * (1.f / TC);
    float d = on ? v - m : 0.f;
    float var = tao_warp_sum(d * d) * (1.f / TC);
    return d * rsqrtf(var + 1e-5f) * g + b;
}

__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }  // nn.GELU
__device__ __forceinline__ float r16(float v) { return __half2float(__float2half(v)); }   // fp16 rounding point

// q|k|v = x Wqkv^T + b, x (N, L, 28) fp16 (T = N L tokens), written into zero-initialised padded (N, 4, L, 8) buffers.
// W = [Wq; Wk; Wv] (3 x 28 x 28), b = [bq | bk | bv], fp32.
extern "C" __global__ void __launch_bounds__(128) tao_qkv(const __half* __restrict__ x, long long T, int L,
        const float* __restrict__ W, const float* __restrict__ b, __half* q, __half* k, __half* v) {
    __shared__ float sw[3 * TC * TC + 3 * TC];
    for (int i = threadIdx.x; i < 3 * TC * TC + 3 * TC; i += blockDim.x) sw[i] = i < 3 * TC * TC ? W[i] : b[i - 3 * TC * TC];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const bool on = lane < TC;
    const float* bb = sw + 3 * TC * TC;
    TAO_WARPS(t, T) {
        float xv = on ? __half2float(x[t * TC + lane]) : 0.f;
        float aq = tao_mv(sw, xv, lane), ak = tao_mv(sw + TC * TC, xv, lane), av = tao_mv(sw + 2 * TC * TC, xv, lane);
        if (on) {
            long long n = t / L;
            int l = (int)(t - n * L);
            long long s = tao_slot(n, L, l * TC + lane);
            q[s] = __float2half(aq + bb[lane]);
            k[s] = __float2half(ak + bb[TC + lane]);
            v[s] = __float2half(av + bb[2 * TC + lane]);
        }
    }
}

// x = LN2(y + W2 gelu(W1 y + b1) + b2), y = LN1(x + Wo o + bo); o = attention output in the padded (N, 4, L, 8)
// layout. In place on x. P = [Wo 28x28 | bo | g1 | n1 | W1 28x28 | b1 | W2 28x28 | b2 | g2 | n2] (fp32).
extern "C" __global__ void __launch_bounds__(128) tao_post(const __half* __restrict__ o, __half* x, long long T, int L,
        const float* __restrict__ P) {
    const int NP = 3 * TC * TC + 7 * TC;
    __shared__ float sp[3 * TC * TC + 7 * TC];
    for (int i = threadIdx.x; i < NP; i += blockDim.x) sp[i] = P[i];
    __syncthreads();
    const float *Wo = sp, *bo = Wo + TC * TC, *g1 = bo + TC, *n1 = g1 + TC, *W1 = n1 + TC, *b1 = W1 + TC * TC,
                *W2 = b1 + TC, *b2 = W2 + TC * TC, *g2 = b2 + TC, *n2 = g2 + TC;
    const int lane = threadIdx.x & 31;
    const bool on = lane < TC;
    const int j = on ? lane : 0;
    TAO_WARPS(t, T) {
        long long n = t / L;
        int l = (int)(t - n * L);
        float a = on ? __half2float(o[tao_slot(n, L, l * TC + lane)]) : 0.f;
        float xv = on ? __half2float(x[t * TC + lane]) : 0.f;
        float y = r16(r16(tao_mv(Wo, a, lane) + bo[j]) + xv);
        y = r16(tao_ln(y, on, g1[j], n1[j]));
        float h = r16(gelu(r16(tao_mv(W1, y, lane) + b1[j])));
        float z = r16(r16(tao_mv(W2, on ? h : 0.f, lane) + b2[j]) + y);
        z = tao_ln(z, on, g2[j], n2[j]);
        if (on) x[t * TC + lane] = __float2half(z);
    }
}
