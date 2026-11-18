#include <cuda_fp16.h>
#include <stdint.h>

// Reproduce the rounded normalized grid before unnormalizing it. Simply
// adding the flow to the integer coordinate diverges at large image widths.
__device__ __forceinline__ float coordinate(int index, int size, float flow) {
    if (size == 1) return 0.f;
    float step = 2.f / (size - 1);
    float base = index < size / 2 ? fmaf(step, index, -1.f) : fmaf(-step, size - index - 1, 1.f);
    float grid = __fadd_rn(base, __fmul_rn(flow, step));
    return fminf(fmaxf(((grid + 1.f) * .5f) * (size - 1), 0.f), float(size - 1));
}

// Both source frames have the same strides. The output stores the pair as
// successive channels-last batches, ready for the next convolutions.
template <typename T>
__device__ void warp_pair(const T* first, const T* second, const float* flow, T* out,
        int batch, int channels, int height, int width, int64_t sn, int64_t sc, int64_t sh, int64_t sw,
        int64_t fn, int64_t fc, int64_t fh, int64_t fw) {
    // One block spans adjacent channels/pixels of a row. The row and batch
    // indices come from the grid; only the within-row index needs division.
    unsigned int channel_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = channel_x / unsigned(channels);
    if (x >= unsigned(width)) return;
    int c = channel_x - x * unsigned(channels);
    for (int pair_n = blockIdx.z; pair_n < 2 * batch; pair_n += gridDim.z) {
        bool right = pair_n >= batch;
        int n = right ? pair_n - batch : pair_n;
        for (int y = blockIdx.y; y < height; y += gridDim.y) {
            T* row = out + (int64_t(pair_n) * height + y) * width * channels;
            const float* f = flow + n * fn + (right ? 2 * fc : 0) + y * fh + x * fw;
            float u = coordinate(x, width, f[0]), v = coordinate(y, height, f[fc]);
            int x0 = int(u), y0 = int(v), x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
            float l = (x0 + 1) - u, r = u - x0, t = (y0 + 1) - v, b = v - y0;
            const T* s = (right ? second : first) + n * sn + c * sc;
            float value = (l * t) * float(s[y0 * sh + x0 * sw]);
            value = fmaf(r * t, float(s[y0 * sh + x1 * sw]), value);
            value = fmaf(l * b, float(s[y1 * sh + x0 * sw]), value);
            row[channel_x] = T(fmaf(r * b, float(s[y1 * sh + x1 * sw]), value));
        }
    }
}

#define WARP_ARGS(T) const T* a, const T* b, const float* f, T* o, int n, int c, int h, int w, \
    int64_t sn, int64_t sc, int64_t sh, int64_t sw, int64_t fn, int64_t fc, int64_t fh, int64_t fw
extern "C" __global__ void warp_f32(WARP_ARGS(float)) {
    warp_pair(a, b, f, o, n, c, h, w, sn, sc, sh, sw, fn, fc, fh, fw);
}
extern "C" __global__ void warp_f16(WARP_ARGS(half)) {
    warp_pair(a, b, f, o, n, c, h, w, sn, sc, sh, sw, fn, fc, fh, fw);
}
