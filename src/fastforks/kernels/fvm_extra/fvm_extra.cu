// Transformer classifier row ops (fast-vision-models `extra.cu`, exp 008). Kernel bodies unchanged; the torch host
// functions became `extern "C"` launchers, the Python wrapper (`__init__.py`) does the checks and allocations.
//  rope_qk  : qkv [B, N, 3*H*D] fp16 (fused GEMM output) -> q, k [B, N, H, D] with rotary embedding
//             (interleaved pairs, timm apply_rot_embed_cat half=False; first npt tokens unrotated)
//  swiglu_ln: gx [M, 2*Hp] = [gate | x] (each padded to Hp >= hid) -> LN(silu(gate) * x) over the first hid channels,
//             [M, Hp] fp16 with zero padding (timm SwiGLU with norm, EVA-02)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void rope_qk_k(const __half* __restrict__ qkv, const float* __restrict__ emb, __half* __restrict__ q,
                          __half* __restrict__ k, int N, int H, int D, int npt, long long total_pairs) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;   // one (token, head, pair) per thread, q and k
    if (i >= total_pairs) return;
    int D2 = D / 2;
    int j = i % D2; long long t = i / D2; int h = t % H; long long tok = t / H;   // tok = b * N + n
    int n = tok % N;
    long long in_q = tok * 3 * H * D + h * D + 2 * j, in_k = in_q + H * D;
    long long o = (tok * H + h) * D + 2 * j;
    __half2 qv = *reinterpret_cast<const __half2*>(qkv + in_q);
    __half2 kv = *reinterpret_cast<const __half2*>(qkv + in_k);
    if (n < npt) {
        *reinterpret_cast<__half2*>(q + o) = qv;
        *reinterpret_cast<__half2*>(k + o) = kv;
        return;
    }
    const float* e = emb + (long long)(n - npt) * 2 * D;
    float s0 = e[2 * j], s1 = e[2 * j + 1], c0 = e[D + 2 * j], c1 = e[D + 2 * j + 1];
    float2 a = __half22float2(qv), b = __half22float2(kv);
    *reinterpret_cast<__half2*>(q + o) = __floats2half2_rn(a.x * c0 - a.y * s0, a.y * c1 + a.x * s1);
    *reinterpret_cast<__half2*>(k + o) = __floats2half2_rn(b.x * c0 - b.y * s0, b.y * c1 + b.x * s1);
}

// qkv [B, N, 3*H*D] fp16 contiguous; emb [N - npt, 2*D] fp32 contiguous; q, k [B, N, H, D] fp16
extern "C" int fvm_rope_qk(const void* qkv, const float* emb, void* q, void* k, int B, int N, int H, int D, int npt,
                           cudaStream_t st) {
    long long total = (long long)B * N * H * (D / 2);
    rope_qk_k<<<(total + 255) / 256, 256, 0, st>>>((const __half*)qkv, emb, (__half*)q, (__half*)k, N, H, D, npt, total);
    return (int)cudaGetLastError();
}

// one block (256 threads) per row
__global__ void swiglu_ln_k(const __half* __restrict__ gx, const __half* __restrict__ w, const __half* __restrict__ b,
                            __half* __restrict__ y, int hid, int Hp, float eps) {
    extern __shared__ float hs[];   // hid floats
    __shared__ float red[2][32];
    long long row = blockIdx.x;
    const __half* g = gx + row * 2 * Hp;
    const __half* x = g + Hp;
    float s = 0.f, s2 = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        float gv = __half2float(g[c]), xv = __half2float(x[c]);
        float h = __half2float(__float2half(gv / (1.f + __expf(-gv)))) * xv;   // fp16 silu output as in eager
        h = __half2float(__float2half(h));
        hs[c] = h; s += h; s2 += h * h;
    }
    for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { red[0][wid] = s; red[1][wid] = s2; }
    __syncthreads();
    if (wid == 0) {
        s = lane < blockDim.x / 32 ? red[0][lane] : 0.f;
        s2 = lane < blockDim.x / 32 ? red[1][lane] : 0.f;
        for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); s2 += __shfl_xor_sync(0xffffffff, s2, o); }
        if (lane == 0) { red[0][0] = s; red[1][0] = s2; }
    }
    __syncthreads();
    float mean = red[0][0] / hid, var = fmaxf(red[1][0] / hid - mean * mean, 0.f), rstd = rsqrtf(var + eps);
    __half* yr = y + row * Hp;
    for (int c = threadIdx.x; c < Hp; c += blockDim.x)
        yr[c] = c < hid ? __float2half((hs[c] - mean) * rstd * __half2float(w[c]) + __half2float(b[c])) : __float2half(0.f);
}

// gx [M, 2*Hp] fp16 contiguous; w, b [hid] fp16; y [M, Hp] fp16
extern "C" int fvm_swiglu_ln(const void* gx, const void* w, const void* b, void* y, long long M, int hid, int Hp, float eps,
                             cudaStream_t st) {
    swiglu_ln_k<<<M, 256, hid * sizeof(float), st>>>((const __half*)gx, (const __half*)w, (const __half*)b, (__half*)y, hid,
                                                     Hp, eps);
    return (int)cudaGetLastError();
}
