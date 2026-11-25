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

// One DySPN step. Read each pixel's five offsets and affinities from the
// propagation convolution, then sample and combine depth without a grid tensor.
__device__ float spn_sample(const float* image, int h, int w, int y, int x, float dy, float dx,
                            float inv_h, float inv_w) {
    float gx = __fmul_rn(__fadd_rn(2.f * dx, float(2 * x - w + 1)), inv_w);
    float gy = __fmul_rn(__fadd_rn(2.f * dy, float(2 * y - h + 1)), inv_h);
    float sx = 0.5f * fmaf(gx + 1.f, float(w), -1.f);
    float sy = 0.5f * fmaf(gy + 1.f, float(h), -1.f);
    if (!(sx > -1.f && sx < w && sy > -1.f && sy < h)) return 0.f;
    int lo_x = __float2int_rd(sx), lo_y = __float2int_rd(sy);
    int hi_x = lo_x + 1, hi_y = lo_y + 1;
    float u0 = hi_x - sx, v0 = hi_y - sy, u1 = sx - lo_x, v1 = sy - lo_y;
    float value = 0.f;
    if (lo_x >= 0 && lo_y >= 0) value = fmaf(image[lo_y * w + lo_x], u0 * v0, value);
    if (hi_x < w && lo_y >= 0) value = fmaf(image[lo_y * w + hi_x], u1 * v0, value);
    if (lo_x >= 0 && hi_y < h) value = fmaf(image[hi_y * w + lo_x], u0 * v1, value);
    if (hi_x < w && hi_y < h) value = fmaf(image[hi_y * w + hi_x], u1 * v1, value);
    return value;
}

extern "C" __global__ void spn_step(const float* image, const float* offsets, const float* missing,
                                   const float* observed, float* out, int h, int w, int step,
                                   int64_t sb, int64_t sc, int64_t sy, int64_t sx, float inv_h, float inv_w,
                                   int64_t count) {
    constexpr int samples = 5, steps = 6;
    constexpr int dx[samples] = {0, -1, 0, 1, 0};
    constexpr int dy[samples] = {-1, 0, 0, 0, 1};
    int area = h * w;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        int p = i % area, y = p / w, x = p % w;
        int64_t batch = i / area;
        const float* src = image + batch * area;
        const float* base = offsets + batch * sb + y * sy + x * sx;
        const float* off = base + 2 * samples * step * sc;
        const float* aff = base + (2 * samples * steps + samples * step) * sc;
        float weights[samples], values[samples], largest = -INFINITY;
        #pragma unroll
        for (int j = 0; j < samples; ++j) {
            weights[j] = aff[j * sc];
            largest = fmaxf(largest, weights[j]);
            values[j] = spn_sample(src, h, w, y, x, off[2 * j * sc] + dy[j], off[(2 * j + 1) * sc] + dx[j],
                                   inv_h, inv_w);
        }
        float denominator = 0.f;
        #pragma unroll
        for (int j = 0; j < samples; ++j) {
            weights[j] = expf(weights[j] - largest);
            denominator += weights[j];
        }
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < samples; ++j) {
            sum = __fadd_rn(sum, __fmul_rn(values[j], __fdividef(weights[j], denominator)));
        }
        out[i] = __fadd_rn(__fmul_rn(missing[i], sum), observed[i]);
    }
}
