#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ float2 blend(float2 a, float2 b, float t) {
    return {(1.f - t) * a.x + t * b.x, (1.f - t) * a.y + t * b.y};
}

// NHWC, channels in pairs. Copy the skip channels directly and write the
// bilinearly upsampled channels alongside them without a full-size temporary.
extern "C" __global__ void resize_cat(const half2* input, const half2* skip, half2* out,
        int h, int w, int oh, int ow, int c, int k, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int channel = i % (c + k);
        int64_t pixel = i / (c + k);
        if (channel >= c) {
            out[i] = skip[pixel * k + channel - c];
            continue;
        }
        float fy = fmaxf(0.f, (float(h) / oh) * (float(pixel / ow % oh) + .5f) - .5f);
        float fx = fmaxf(0.f, (float(w) / ow) * (float(pixel % ow) + .5f) - .5f);
        int y0 = int(fy), x0 = int(fx), x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
        int64_t batch = pixel / ow / oh;
        const half2* base = input + batch * h * w * c + channel;
        float2 a = blend(__half22float2(base[(int64_t(y0)*w+x0)*c]), __half22float2(base[(int64_t(y0)*w+x1)*c]), fx - x0);
        float2 b = blend(__half22float2(base[(int64_t(y1)*w+x0)*c]), __half22float2(base[(int64_t(y1)*w+x1)*c]), fx - x0);
        float2 result = blend(a, b, fy - y0);
        out[i] = __floats2half2_rn(result.x, result.y);
    }
}
