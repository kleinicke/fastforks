#include <cuda_fp16.h>

// The aligned-corner NHWC interpolation from dpt_fused, using source rows at the full output scale.
extern "C" __global__ void bilinear_rows(const __half* __restrict__ src, __half* __restrict__ dst,
                              int Hs, int Ws, int hd, int Wd, int C, int first,
                              float sh, float sw, long long total) {
    const int CV = C / 8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd);
        long long r = pix / Wd;
        int y = first + (int)(r % hd), n = (int)(r / hd);
        float fy = sh * y;
        float fx = sw * x;
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < Hs - 1), x1 = x0 + (x0 < Ws - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* s = src + (long long)n * Hs * Ws * C + c0;
        uint4 a = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x0) * C));
        uint4 b = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x1) * C));
        uint4 c = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x0) * C));
        uint4 d = __ldg(reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x1) * C));
        const __half *a8 = reinterpret_cast<const __half*>(&a), *b8 = reinterpret_cast<const __half*>(&b);
        const __half *c8 = reinterpret_cast<const __half*>(&c), *d8 = reinterpret_cast<const __half*>(&d);
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o8[k] = __float2half(hy * (hx * __half2float(a8[k]) + lx * __half2float(b8[k])) +
                                 ly * (hx * __half2float(c8[k]) + lx * __half2float(d8[k])));
        *reinterpret_cast<uint4*>(dst + pix * C + c0) = o;
    }
}
