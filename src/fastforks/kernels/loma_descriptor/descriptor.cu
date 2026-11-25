#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// Sample the penultimate decoder map at four final-resolution pixel centres.
// The resize rounds to fp16 before adding the last-stage projection. Only the
// final keypoint interpolation is accumulated in fp32.
extern "C" __global__ void descriptors(const half* prior, const half* delta, const int64_t* xs,
        const int64_t* ys, const float* weights, float* out, int n, int c, int ph, int pw,
        float scale_y, float scale_x, int64_t pc, int64_t py, int64_t px,
        int64_t dc, int64_t dk, int64_t dn, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int point = i / c, ch = i % c;
        const half* plane = prior + ch * pc;
        float sum = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int64_t j = int64_t(k) * n + point;
            float x = fmaxf(0.f, (float(xs[j]) + .5f) * scale_x - .5f);
            float y = fmaxf(0.f, (float(ys[j]) + .5f) * scale_y - .5f);
            int left = int(x), top = int(y), right = min(left + 1, pw - 1), bottom = min(top + 1, ph - 1);
            float fx = x - left, fy = y - top;
            float upper = (1.f-fx)*__half2float(plane[top*py+left*px]) + fx*__half2float(plane[top*py+right*px]);
            float lower = (1.f-fx)*__half2float(plane[bottom*py+left*px]) + fx*__half2float(plane[bottom*py+right*px]);
            half resized = __float2half_rn((1.f-fy)*upper + fy*lower);
            half value = __hadd(delta[ch*dc + k*dk + point*dn], resized);
            sum += __half2float(value) * weights[j];
        }
        out[i] = sum;
    }
}
