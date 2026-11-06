// MoCha-Stereo kernels path (NHWC fp16): the fused LSTM gate update after the merged 4-gate convolution. Compiled
// once per target (tools/build_kernels.py), launched on torch's current stream. Source: our MoCha-Stereo fork
// (08b4e31, core/igev_kernels.cu lstm_gates).
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

// g, ctx: [P, 4 Hd] raw gate pre-activations (i | f | c~ | o) and context (+ gate biases). c: fp32 [P, Hd] cell state,
// or null: the cell input is the previous h (upstream's aliased netC / netH lists). h: [P, hs] fp16, updated in place.
extern "C" __global__ void lstm_gates(const __half* __restrict__ g, const __half* __restrict__ ctx, float* c,
                                      __half* h, int hs, long long P, int Hd) {
    GRID_STRIDE(i, P * Hd) {
        long long p = i / Hd;
        int k = (int)(i - p * Hd);
        long long b = p * 4 * Hd + k;
        float gi = __half2float(g[b]) + __half2float(ctx[b]);
        float gf = __half2float(g[b + Hd]) + __half2float(ctx[b + Hd]);
        float gc = __half2float(g[b + 2 * Hd]) + __half2float(ctx[b + 2 * Hd]);
        float go = __half2float(g[b + 3 * Hd]) + __half2float(ctx[b + 3 * Hd]);
        float cp = c ? c[i] : __half2float(h[p * hs + k]);
        float cn = cp / (1.f + __expf(-gf)) + tanhf(gc) / (1.f + __expf(-gi));
        if (c) c[i] = cn;
        h[p * hs + k] = __float2half(tanhf(cn) / (1.f + __expf(-go)));
    }
}
