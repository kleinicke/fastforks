#include <stdint.h>
#include <math.h>

template<typename T>
__device__ void resize(const float* src, const int64_t* xs, const T* wx, const int64_t* ys,
                      const T* wy, float* dst, int h, int w, int oh, int ow, int nx, int ny,
                      int rounding, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const int64_t col = (i % ow) * nx;
        const int64_t row = ((i / ow) % oh) * ny;
        const float* plane = src + (i / ow / oh) * h * w;
        T accum = 0;
        for (int y = 0; y < ny; ++y) {
            const float* line = plane + ys[row + y] * w;
            T horizontal = 0;
            for (int x = 0; x < nx; ++x) horizontal += T(line[xs[col + x]]) * wx[col + x];
            accum += horizontal * wy[row + y];
        }
        if (rounding) accum = min(T(255), max(T(0), rounding == 1 ? nearbyint(accum) : floor(accum + T(0.5))));
        dst[i] = float(accum);
    }
}

#define RESIZE(name, T) extern "C" __global__ void name(const float* src, const int64_t* xs, const T* wx, \
    const int64_t* ys, const T* wy, float* dst, int h, int w, int oh, int ow, int nx, int ny, int rounding, int64_t n) { \
    resize(src, xs, wx, ys, wy, dst, h, w, oh, ow, nx, ny, rounding, n); }

RESIZE(cv_cubic, double)
RESIZE(cv_area, float)

extern "C" __global__ void u8_levels(const float* src, float* dst, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x)
        dst[i] = fminf(255.f, fmaxf(0.f, nearbyintf(src[i] * 255.f)));
}
