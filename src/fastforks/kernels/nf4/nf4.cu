// NF4 (bitsandbytes 4-bit NormalFloat) -> bf16 dequantization, fused table lookup + scale + round.
//
// q: packed codes, two per byte (high nibble first), read as 64-bit words (8 bytes = 16 weights);
// absmax: one bf16 scale per 64 weights (= per 4 words); code: the 16 code values (fp32);
// out: bf16 weights, written as 64-bit words (4 weights each). Rounding is fp32 -> bf16 round to nearest
// even, i.e. `(code * absmax).to(bfloat16)` as bitsandbytes' `dequantize_4bit` computes it.
extern "C" __global__ void nf4_dequant(const unsigned long long* __restrict__ q,
                                       const unsigned short* __restrict__ absmax,
                                       const float* __restrict__ code,
                                       unsigned long long* __restrict__ out, long long n_words) {
    float c[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) c[j] = code[j];
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n_words; i += stride) {
        const float am = __uint_as_float(((unsigned)absmax[i >> 2]) << 16);
        const unsigned long long w = q[i];
#pragma unroll
        for (int g = 0; g < 4; ++g) {  // 2 bytes -> 4 weights -> one 64-bit store
            unsigned long long o = 0;
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const unsigned byte = (unsigned)(w >> (16 * g + 8 * b)) & 255u;
                unsigned hi = __float_as_uint(c[byte >> 4] * am), lo = __float_as_uint(c[byte & 15u] * am);
                hi = (hi + 0x7FFFu + ((hi >> 16) & 1u)) >> 16;  // fp32 -> bf16, round to nearest even
                lo = (lo + 0x7FFFu + ((lo >> 16) & 1u)) >> 16;
                o |= ((unsigned long long)(hi | (lo << 16))) << (32 * b);
            }
            out[4 * i + g] = o;
        }
    }
}

// LoRA epilogues of an NF4 linear: `y` (bf16 GEMM + bias) and `l` (bf16 LoRA term) are summed in fp32 as upstream's
// autocast does (`y.float() + l`), then consumed in the same pass. All tensors contiguous, 8 elements (16 bytes of
// bf16) per step; `n8` = elements / 8, `N8` = row length / 8 for the per-column vectors. Every op is IEEE round to
// nearest (no contraction), so the results equal the PyTorch chains they replace bit for bit.
#include <cuda_bf16.h>

__device__ __forceinline__ void unpack8(uint4 v, float* f) {
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        f[2 * k] = __uint_as_float(w[k] << 16);
        f[2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
    }
}

__device__ __forceinline__ unsigned pack2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}

__device__ __forceinline__ void sum8(const uint4* y, const uint4* l, long long i, float* s) {
    float b[8];
    unpack8(y[i], s);
    unpack8(l[i], b);
#pragma unroll
    for (int k = 0; k < 8; ++k) s[k] = __fadd_rn(s[k], b[k]);
}

#define GRID_LOOP(i, n) \
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

// out (fp32) = y + l
extern "C" __global__ void nf4_lora_f32(const uint4* __restrict__ y, const uint4* __restrict__ l,
                                        float4* __restrict__ out, long long n8) {
    GRID_LOOP(i, n8) {
        float s[8];
        sum8(y, l, i, s);
        out[2 * i] = make_float4(s[0], s[1], s[2], s[3]);
        out[2 * i + 1] = make_float4(s[4], s[5], s[6], s[7]);
    }
}

// out (bf16) = bf16(y + l)
extern "C" __global__ void nf4_lora_bf16(const uint4* __restrict__ y, const uint4* __restrict__ l,
                                         uint4* __restrict__ out, long long n8) {
    GRID_LOOP(i, n8) {
        float s[8];
        sum8(y, l, i, s);
        out[i] = pack8(s);
    }
}

// out (bf16) = bf16(gelu_tanh(y + l)), PyTorch's tanh-GELU formula in fp32
extern "C" __global__ void nf4_lora_gelu_bf16(const uint4* __restrict__ y, const uint4* __restrict__ l,
                                              uint4* __restrict__ out, long long n8) {
    const float kBeta = (float)(M_SQRT2 * M_2_SQRTPI * 0.5), kKappa = 0.044715f;
    GRID_LOOP(i, n8) {
        float s[8];
        sum8(y, l, i, s);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float x = s[k], x3 = x * x * x;
            s[k] = 0.5f * x * (1.0f + tanhf(kBeta * (x + kKappa * x3)));
        }
        out[i] = pack8(s);
    }
}

// res (fp32, in place) = (y + l) * gate[col] + res   (gate bf16 [N])
extern "C" __global__ void nf4_lora_gate_res(const uint4* __restrict__ y, const uint4* __restrict__ l,
                                             const uint4* __restrict__ gate, float4* __restrict__ res, long long n8,
                                             long long N8) {
    GRID_LOOP(i, n8) {
        float s[8], g[8];
        sum8(y, l, i, s);
        unpack8(gate[i % N8], g);
        const float4 r0 = res[2 * i], r1 = res[2 * i + 1];
        const float r[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) s[k] = __fadd_rn(__fmul_rn(s[k], g[k]), r[k]);
        res[2 * i] = make_float4(s[0], s[1], s[2], s[3]);
        res[2 * i + 1] = make_float4(s[4], s[5], s[6], s[7]);
    }
}

// out (bf16) = bf16(x * scale[col] + shift[col])   (x fp32, scale / shift bf16 [N]): the block's modulation
extern "C" __global__ void nf4_mod_bf16(const float4* __restrict__ x, const uint4* __restrict__ scale,
                                        const uint4* __restrict__ shift, uint4* __restrict__ out, long long n8,
                                        long long N8) {
    GRID_LOOP(i, n8) {
        float s[8], b[8];
        unpack8(scale[i % N8], s);
        unpack8(shift[i % N8], b);
        const float4 x0 = x[2 * i], x1 = x[2 * i + 1];
        const float v[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) s[k] = __fadd_rn(__fmul_rn(v[k], s[k]), b[k]);
        out[i] = pack8(s);
    }
}

// out[t0 + t] (bf16 [T, H, D]) = bf16(rope(x * rsqrt(v + 1e-6) * w, f[t0 + t]))   (x fp32 [Ts, H, D], v fp32
// [Ts, H] = mean(x^2), w fp32 [D], f complex64 [T, D / 2]): RMS-normed, rotated q / k of one stream, written into
// its token range of the joint sequence. One thread per (re, im) pair; the complex product contracts as PyTorch's
// (c10::complex) does, so the result is bit-equal to the PyTorch chain.
extern "C" __global__ void nf4_rms_rope_bf16(const float2* __restrict__ x, const float* __restrict__ v,
                                             const float2* __restrict__ w, const float2* __restrict__ f,
                                             __nv_bfloat162* __restrict__ out, long long n2, int heads, int half,
                                             long long t0) {
    GRID_LOOP(p, n2) {
        const long long row = p / half, t = row / heads;
        const int d = (int)(p % half);
        const float r = rsqrtf(__fadd_rn(v[row], 1e-6f));
        const float2 xv = x[p], wv = w[d], fv = f[(t0 + t) * half + d];
        const float a = __fmul_rn(__fmul_rn(xv.x, r), wv.x), b = __fmul_rn(__fmul_rn(xv.y, r), wv.y);
        const float re = __fmaf_rn(a, fv.x, -__fmul_rn(b, fv.y)), im = __fmaf_rn(b, fv.x, __fmul_rn(a, fv.y));
        out[t0 * heads * half + p] = __floats2bfloat162_rn(re, im);
    }
}
