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
