#include <cuda_runtime.h>
#include <stdint.h>

constexpr float weight = 1.0f / 3.0f;

// Store one weighted (u,v) edge pair for each pooled cell. The final column
// and row have no outgoing horizontal and vertical edge, respectively.
template<int R>
__device__ float2 edges(const float* image, int y, int x, int h, int w) {
    float u = 0.f, v = 0.f;
    for (int j = 0; j < R; ++j) {
        for (int i = 0; i < R; ++i) {
            int k = (y * R + j) * w + x * R + i;
            if ((x + 1) * R < w) u += image[k + R] - image[k];
            if ((y + 1) * R < h) v += image[k + R * w] - image[k];
        }
    }
    // Pool mean, gradient confidence, then the transpose broadcasts / R^2.
    return make_float2(((u / (R * R)) * weight) / (R * R),
                       ((v / (R * R)) * weight) / (R * R));
}

extern "C" __global__ void pool_edges(const float* image, float2* out, int h, int w, int64_t count) {
    const int level2 = h / 2 * (w / 2), cells = level2 + h / 4 * (w / 4);
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const float* in = image + (i / cells) * h * w;
        int j = i % cells;
        if (j < level2) out[i] = edges<2>(in, j / (w / 2), j % (w / 2), h, w);
        else {
            j -= level2;
            out[i] = edges<4>(in, j / (w / 4), j % (w / 4), h, w);
        }
    }
}

__device__ float divergence(const float2* edges, int y, int x, int width) {
    int i = y * width + x;
    float2 f = edges[i];
    float left = x ? edges[i - 1].x : 0.f;
    float top = y ? edges[i - width].y : 0.f;
    return ((left - f.x) + top) - f.y;
}

extern "C" __global__ void normal(const float* image, const float* diag, const float2* coarse,
                                  float* out, int h, int w, int64_t count) {
    const int level2 = h / 2 * (w / 2), cells = level2 + h / 4 * (w / 4);
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int x = i % w, y = (i / w) % h;
        const float2* pooled = coarse + (i / w / h) * cells;
        float v = image[i];
        float left = x ? __fmul_rn(v - image[i - 1], weight) : 0.f;
        float right = x + 1 < w ? __fmul_rn(image[i + 1] - v, weight) : 0.f;
        float top = y ? __fmul_rn(v - image[i - w], weight) : 0.f;
        float bottom = y + 1 < h ? __fmul_rn(image[i + w] - v, weight) : 0.f;
        float sum = ((left - right) + top) - bottom;
        sum += divergence(pooled, y / 2, x / 2, w / 2);
        sum += divergence(pooled + level2, y / 4, x / 4, w / 4);
        out[i] = sum + __fmul_rn(diag[i], v);
    }
}
