#include <cuda_fp16.h>
#include <stdint.h>

// Normalize each pixel-centered triangular filter over the available input.
// Both axes share one bounds table and one flat arena of filter coefficients.
extern "C" __global__ void triangle_tables(int2* bounds, float* weights,
        int height, int width, int side, int ystride, int xstride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * side) return;
    bool horizontal = i >= side;
    int input = horizontal ? width : height;
    int pos = horizontal ? i - side : i;
    int step = horizontal ? xstride : ystride;
    float scale = float(input) / side;
    float support = fmaxf(scale, 1.f);
    float center = scale * (pos + 0.5f);
    int first = max(0, int((center - support) + 0.5f));
    int stop = min(input, int((center + support) + 0.5f));
    bounds[i] = make_int2(first, stop - first);
    float* w = weights + (horizontal ? int64_t(side) * ystride : 0) + int64_t(pos) * step;
    // The CUDA resize evaluates first - center as a fused multiply-add.
    // Rounding center alone moves source pixels across half-integer output ties.
    float offset = fmaf(-scale, pos + 0.5f, float(first));
    float inv = 1.f / support;
    float total = 0.f;
    for (int k = 0; k < stop - first; ++k) {
        float d = ((k + offset) + 0.5f) * inv;
        w[k] = fmaxf(0.f, 1.f - fabsf(d));
        total += w[k];
    }
    for (int k = 0; k < stop - first; ++k) w[k] /= total;
}

template <typename T>
__device__ __forceinline__ float level(T v);
template <> __device__ __forceinline__ float level<unsigned char>(unsigned char v) { return float(v); }
template <> __device__ __forceinline__ float level<float>(float v) { return rintf(v * 255.f); }

template <typename T>
__device__ void resize_rgb(const T* image, const int2* bounds, const float* weights, half* output,
        int side, int ystride, int xstride, int64_t sb, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        int x = i % side, y = (i / side) % side;
        int64_t plane = i / (int64_t(side) * side);
        int2 yr = bounds[y], xr = bounds[side+x];
        const float* wy = weights + int64_t(y) * ystride;
        const float* wx = weights + int64_t(side) * ystride + int64_t(x) * xstride;
        const T* p = image + (plane/3)*sb + (plane%3)*sc + int64_t(yr.x)*sy + int64_t(xr.x)*sx;
        float value = 0.f;
        for (int v = 0; v < yr.y; ++v, p += sy) {
            float row = 0.f;
            for (int u = 0; u < xr.y; ++u) row = fmaf(level(p[int64_t(u)*sx]), wx[u], row);
            value = fmaf(row, wy[v], value);
        }
        float quantized = fminf(255.f, fmaxf(0.f, rintf(value)));
        output[i] = __float2half_rn((quantized / 255.f - 0.5f) * 2.f);
    }
}

extern "C" __global__ void resize_rgb_u8(const unsigned char* image, const int2* bounds, const float* weights,
        half* output, int side, int ystride, int xstride, int64_t sb, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    resize_rgb(image,bounds,weights,output,side,ystride,xstride,sb,sc,sy,sx,count);
}
extern "C" __global__ void resize_rgb_f32(const float* image, const int2* bounds, const float* weights,
        half* output, int side, int ystride, int xstride, int64_t sb, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    resize_rgb(image,bounds,weights,output,side,ystride,xstride,sb,sc,sy,sx,count);
}
