#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void resize_normalize(const float* image, const float* mean, const float* stddev,
    half* out, int height, int width, int side, int64_t sb, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    const float scale_y = float(height) / side, scale_x = float(width) / side;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int ox = i % side, oy = (i / side) % side;
        int64_t bc = i / side / side;
        int channel = bc % 3;
        float y = fmaxf(__fmaf_rn(scale_y, oy + 0.5f, -0.5f), 0.f);
        float x = fmaxf(__fmaf_rn(scale_x, ox + 0.5f, -0.5f), 0.f);
        int y0 = min(int(y), height - 1), x0 = min(int(x), width - 1);
        int y1 = min(y0 + 1, height - 1), x1 = min(x0 + 1, width - 1);
        float fy = y - y0, fx = x - x0;
        const float* src = image + (bc / 3) * sb + channel * sc;
        // Match the low-index multiply-add contraction used by ATen CUDA.
        float row0 = __fmaf_rn(1.f - fx, src[y0 * sy + x0 * sx], __fmul_rn(fx, src[y0 * sy + x1 * sx]));
        float row1 = __fmaf_rn(1.f - fx, src[y1 * sy + x0 * sx], __fmul_rn(fx, src[y1 * sy + x1 * sx]));
        float value = __fmaf_rn(1.f - fy, row0, __fmul_rn(fy, row1));
        out[i] = __float2half((value - mean[channel]) / stddev[channel]);
    }
}
