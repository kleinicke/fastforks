#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// Match grid_sample's align_corners=False coordinates, zero padding and
// accumulation order. The following correlation consumes the samples in fp16.
template <typename T>
__device__ inline float raster(float pixel, int size, int offset) {
    const float cell = pixel * (size / 256.f) + offset;
    float normalized = 2.f * (cell / size) - 1.f;
    if constexpr (sizeof(T) == sizeof(half))
        normalized = __half2float(__float2half_rn(normalized));
    return ((normalized + 1.f) * size - 1.f) * .5f;
}

template <typename T>
__device__ void sample(const T* grid, const float* points, half* output,
                       int t, int h, int w, int c, int n,
                       int64_t sb, int64_t st, int64_t sy, int64_t sx, int64_t sc, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int channel = i % c;
        const int64_t site = i / c;
        const int offset = site % 49;
        const int64_t point = site / 49;
        const float x = raster<T>(points[2 * point], w, offset % 7 - 3);
        const float y = raster<T>(points[2 * point + 1], h, offset / 7 - 3);
        float sum = 0.f;
        if (x > -1.f && x < w && y > -1.f && y < h) {
            const int ix = __float2int_rd(x), iy = __float2int_rd(y);
            const T* frame = grid + point / t / n * sb + point % t * st + channel * sc;
            #pragma unroll
            for (int dy = 0; dy < 2; ++dy) {
                #pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                    if (unsigned(ix + dx) < unsigned(w) && unsigned(iy + dy) < unsigned(h)) {
                        const float wx = dx ? x - ix : ix + 1 - x;
                        const float wy = dy ? y - iy : iy + 1 - y;
                        sum = fmaf(float(frame[(iy + dy) * sy + (ix + dx) * sx]), wx * wy, sum);
                    }
                }
            }
        }
        output[i] = __float2half_rn(sum);
    }
}

#define ENTRY(name, T) \
extern "C" __global__ void name(const T* grid, const float* points, half* output, \
        int t, int h, int w, int c, int n, int64_t sb, int64_t st, int64_t sy, \
        int64_t sx, int64_t sc, int64_t count) { \
    sample(grid, points, output, t, h, w, c, n, sb, st, sy, sx, sc, count); \
}

ENTRY(sample_float, float)
ENTRY(sample_half, half)
