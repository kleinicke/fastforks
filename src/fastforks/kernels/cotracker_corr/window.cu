#include <cuda_fp16.h>
#include <stdint.h>
#include <assert.h>

__device__ int64_t frame_index(int64_t v, int n) {
    int64_t i = v < 0 ? v + n : v;
    assert(i >= 0 && i < n);
    return i;
}

extern "C" __global__ void gather(const half* feature, const int64_t* frames, const float* coords,
        half* out, float* fraction, int t, int h, int w, int channels, int64_t count) {
    for (int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int c = i % channels;
        int64_t p = i / channels;
        int y = p % 8, x = (p / 8) % 8;
        int64_t k = p / 64;
        float cx = coords[2*k], cy = coords[2*k+1];
        float x0 = floorf(cx), y0 = floorf(cy);
        if (!(x || y || c)) {
            fraction[2*k] = cx-x0;
            fraction[2*k+1] = cy-y0;
        }
        int sx = int(fminf(fmaxf(x0 + x - 3, 0.f), float(w-1)));
        int sy = int(fminf(fmaxf(y0 + y - 3, 0.f), float(h-1)));
        out[i] = feature[((frame_index(frames[k], t) * h + sy) * w + sx) * channels + c];
    }
}

__device__ float mix(float a, float b, float fraction) {
    float delta = b - a;
    return fraction < .5f ? fmaf(fraction, delta, a) : fmaf(-(1.f - fraction), delta, b);
}

extern "C" __global__ void interpolate(const half* g, const float* fraction, half* out,
        int channels, int64_t row_stride, int64_t count) {
    for (int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int c = i % channels;
        int64_t p = i / channels;
        int offset = p % 49, x = offset / 7, y = offset % 7;
        int64_t k = p / 49;
        const half* f = g + (k*64+x*8+y)*channels+c;
        float fx = __half2float(__float2half(fraction[2*k]));
        float fy = __half2float(__float2half(fraction[2*k+1]));
        // torch.lerp rounds each vertical mixture to half before the horizontal pass.
        float a = __half2float(__float2half(mix(__half2float(f[0]), __half2float(f[channels]), fy)));
        float b = __half2float(__float2half(mix(__half2float(f[8*channels]), __half2float(f[9*channels]), fy)));
        out[k*row_stride + offset*channels + c] = __float2half(mix(a, b, fx));
    }
}

extern "C" __global__ void support(const half* feature, const int64_t* frames, const float* coords,
        half* out, int t, int h, int w, int channels, int64_t count) {
    for (int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int offset = i % 49, c = (i / 49) % channels;
        int64_t k = i / 49 / channels;
        float cx = coords[2*k], cy = coords[2*k+1];
        float bx = floorf(cx), by = floorf(cy);
        int x = offset / 7 - 3, y = offset % 7 - 3;
        int x0 = int(fminf(fmaxf(bx+x, 0.f), float(w-1)));
        int x1 = int(fminf(fmaxf(bx+x+1, 0.f), float(w-1)));
        int y0 = int(fminf(fmaxf(by+y, 0.f), float(h-1)));
        int y1 = int(fminf(fmaxf(by+y+1, 0.f), float(h-1)));
        const half* f = feature + frame_index(frames[k],t)*h*w*channels + c;
        float a = mix(__half2float(f[(y0*w+x0)*channels]), __half2float(f[(y1*w+x0)*channels]), cy-by);
        float b = mix(__half2float(f[(y0*w+x1)*channels]), __half2float(f[(y1*w+x1)*channels]), cy-by);
        out[i] = __float2half(mix(a,b,cx-bx));
    }
}
