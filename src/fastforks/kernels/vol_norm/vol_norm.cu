// Instance norm (affine) + LeakyReLU of channels-last 3D fp16 activations, in place: nnU-Net's conv block epilogue.
//
// x: [N, S, C] fp16 (channels_last_3d, S = D * H * W), C % 8 == 0, 16-byte aligned. Three launches:
//   in_stats_h: per (n, c) shifted sums  sum(x - k), sum((x - k)^2)  in fp32 (k = x[n, 0, c], against cancellation),
//               each block writes its own shifted moments into [N, blocks, C, 2].
//   in_final_h: scale = w * rsqrt(var + eps), shift = b - mean * scale  (biased variance, as InstanceNorm3d) into
//               stats, in place.
//   in_apply_h: y = x * scale + shift (fp32), y = y >= 0 ? y : slope * y, stored fp16; 8 channels per thread.
#include <cuda_fp16.h>
#include <stdint.h>

union H8 {
    uint4 u;
    __half2 h[4];
};

// grid (blocks, N), block (C / 8, rows); shared rows * C * 2 floats
extern "C" __global__ void in_stats_h(const __half* __restrict__ x, float* __restrict__ stats, int S, int C, const __half* conv_bias) {
    extern __shared__ float sh[];
    const int c8 = threadIdx.x, C8 = blockDim.x, n = blockIdx.y;
    const __half* xn = x + (int64_t)n * S * C;
    H8 k;
    k.u = *reinterpret_cast<const uint4*>(xn + c8 * 8);
    H8 bias;
    if (conv_bias) bias.u = __ldg(reinterpret_cast<const uint4*>(conv_bias) + c8);
    float kf[8], s1[8], s2[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __half22float2(conv_bias ? __hadd2(k.h[j], bias.h[j]) : k.h[j]);
        kf[2 * j] = f.x;
        kf[2 * j + 1] = f.y;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) s1[j] = s2[j] = 0.f;
    for (int64_t r = (int64_t)blockIdx.x * blockDim.y + threadIdx.y; r < S; r += (int64_t)gridDim.x * blockDim.y) {
        H8 v;
        v.u = __ldg(reinterpret_cast<const uint4*>(xn + r * C + c8 * 8));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(conv_bias ? __hadd2(v.h[j], bias.h[j]) : v.h[j]);
            float a = f.x - kf[2 * j], b = f.y - kf[2 * j + 1];
            s1[2 * j] += a;
            s2[2 * j] = fmaf(a, a, s2[2 * j]);
            s1[2 * j + 1] += b;
            s2[2 * j + 1] = fmaf(b, b, s2[2 * j + 1]);
        }
    }
    float* row = sh + (threadIdx.y * C8 + c8) * 16;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        row[j] = s1[j];
        row[8 + j] = s2[j];
    }
    __syncthreads();
    // threads of the block sum one (channel, moment) column each over the rows
    const int t = threadIdx.y * C8 + c8, nt = C8 * blockDim.y;
    for (int q = t; q < C * 2; q += nt) {
        int c = q >> 1, m = q & 1;  // channel c = c8' * 8 + j
        int cc8 = c >> 3, j = c & 7;
        float acc = 0.f;
        for (int y = 0; y < blockDim.y; ++y) acc += sh[(y * C8 + cc8) * 16 + m * 8 + j];
        stats[(((int64_t)n * gridDim.x + blockIdx.x) * C + c) * 2 + m] = acc;
    }
}

// Reduce the block moments in a fixed order. Adjacent lanes read adjacent
// channels and eight cooperating warps cover the spatial block axis.
extern "C" __global__ void in_final_h(const __half* __restrict__ x, const float2* __restrict__ partial,
                                      float2* __restrict__ stats, const __half* __restrict__ w,
                                      const __half* __restrict__ b, int blocks, int S, int C, float eps, const __half* conv_bias) {
    __shared__ float2 sums[8][32];
    int n = blockIdx.y, c = blockIdx.x * 32 + threadIdx.x;
    float2 sum = make_float2(0.f, 0.f);
    if (c < C) {
        for (int i = threadIdx.y; i < blocks; i += 8) {
            float2 v = partial[((int64_t)n * blocks + i) * C + c];
            sum.x += v.x;
            sum.y += v.y;
        }
    }
    sums[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y || c >= C) return;
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        sum.x += sums[i][threadIdx.x].x;
        sum.y += sums[i][threadIdx.x].y;
    }
    __half first = x[(int64_t)n * S * C + c];
    float k = __half2float(conv_bias ? __hadd(first, conv_bias[c]) : first);
    float inv = 1.f / (float)S;
    float d = sum.x * inv;
    float var = fmaxf(fmaf(-d, d, sum.y * inv), 0.f);
    float scale = __half2float(w[c]) * rsqrtf(var + eps);
    stats[(int64_t)n*C+c] = make_float2(scale, fmaf(-(k + d), scale, __half2float(b[c])));
}

// N * S * C / 8 grid-stride items
extern "C" __global__ void in_apply_h(__half* __restrict__ x, const float* __restrict__ stats, int64_t total, int S,
                                      int C, float slope, const __half* conv_bias) {
    const int C8 = C >> 3;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int c8 = (int)(i % C8);
        int n = (int)(i / ((int64_t)S * C8));
        const float4* st = reinterpret_cast<const float4*>(stats + ((int64_t)n * C + c8 * 8) * 2);
        H8 bias;
        if (conv_bias) bias.u = __ldg(reinterpret_cast<const uint4*>(conv_bias) + c8);
        H8 v;
        v.u = reinterpret_cast<uint4*>(x)[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 p = __ldg(st + j);  // scale, shift of channels 2j, 2j+1
            float2 f = __half22float2(conv_bias ? __hadd2(v.h[j], bias.h[j]) : v.h[j]);
            float a = fmaf(f.x, p.x, p.y), bb = fmaf(f.y, p.z, p.w);
            a = a >= 0.f ? a : a * slope;
            bb = bb >= 0.f ? bb : bb * slope;
            v.h[j] = __floats2half2_rn(a, bb);
        }
        reinterpret_cast<uint4*>(x)[i] = v.u;
    }
}
