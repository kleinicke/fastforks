#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// Cooperatively transpose a tile of NHWC pixels into shared memory. One
// warp per output channel then accumulates its 128 weights in channel order.
extern "C" __global__ void dense_project(const half *features, const float *weight, const float *bias,
                                         float *output, int64_t pixels) {
    __shared__ half tile[32][130]; // padding avoids bank conflicts when a warp reads one channel
    int lane = threadIdx.x & 31, head = threadIdx.x >> 5;
    for (int64_t first = int64_t(blockIdx.x) * 32; first < pixels; first += int64_t(gridDim.x) * 32) {
        for (int i = threadIdx.x; i < 32 * 128; i += blockDim.x) {
            int row = i / 128, channel = i % 128;
            if (first + row < pixels)
                tile[row][channel] = features[(first + row) * 128 + channel];
        }
        __syncthreads();
        int64_t pixel = first + lane;
        if (pixel < pixels) {
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < 128; ++c)
                sum = fmaf(__half2float(tile[lane][c]), weight[head * 128 + c], sum);
            output[pixel * 6 + head] = sum + bias[head];
        }
        __syncthreads();
    }
}

__device__ float3 diff(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

// PyTorch's length-three reduction has a different pair order when the
// coordinate is the fast dimension. Each squared component rounds first.
__device__ float reduce3(float a, float b, float c, bool packed) {
    return packed ? __fadd_rn(__fadd_rn(a, c), b) : __fadd_rn(__fadd_rn(a, b), c);
}

__device__ float3 unit(float3 a, bool packed) {
    float norm = sqrtf(reduce3(__fmul_rn(a.x, a.x), __fmul_rn(a.y, a.y), __fmul_rn(a.z, a.z), packed)) + 1e-12f;
    return make_float3(a.x / norm, a.y / norm, a.z / norm);
}

__device__ float3 triangle(float3 a, float3 b, bool packed) {
    return unit(make_float3(fmaf(a.y, b.z, -__fmul_rn(a.z, b.y)), fmaf(a.z, b.x, -__fmul_rn(a.x, b.z)),
                            fmaf(a.x, b.y, -__fmul_rn(a.y, b.x))),
                packed);
}

__device__ float3 point(const float *base, int y, int x, int64_t sy, int64_t sx, int64_t sc) {
    const float *p = base + y * sy + x * sx;
    return make_float3(p[0], p[sc], p[2 * sc]);
}

// xyzw stores the twice-normalized XYZ normal and whether any incident
// triangle was valid. Model points can be either planar or interleaved.
extern "C" __global__ void normals(const float *points, const bool *mask, float4 *output, int height, int width,
                                   int64_t pixels, int64_t sn, int64_t sy, int64_t sx, int64_t sc) {
    for (int64_t pos = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; pos < pixels;
         pos += int64_t(gridDim.x) * blockDim.x) {
        int x = pos % width, y = (pos / width) % height;
        const float *base = points + (pos / (int64_t(height) * width)) * sn;
        float3 center = point(base, y, x, sy, sx, sc);
        float3 zero = make_float3(0, 0, 0);
        float3 neighbor[4] = {diff(y ? point(base, y - 1, x, sy, sx, sc) : zero, center),
                              diff(x ? point(base, y, x - 1, sy, sx, sc) : zero, center),
                              diff(y + 1 < height ? point(base, y + 1, x, sy, sx, sc) : zero, center),
                              diff(x + 1 < width ? point(base, y, x + 1, sy, sx, sc) : zero, center)};
        bool good[4] = {y && mask[pos - width], x && mask[pos - 1], y + 1 < height && mask[pos + width],
                        x + 1 < width && mask[pos + 1]};
        float3 sum = zero;
        bool valid = false;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            bool keep = mask[pos] && good[i] && good[j];
            float3 n = triangle(neighbor[i], neighbor[j], true);
            sum.x += n.x * float(keep);
            sum.y += n.y * float(keep);
            sum.z += n.z * float(keep);
            valid |= keep;
        }
        float3 n = valid ? unit(unit(sum, sc == 1), sc == 1) : zero;
        output[pos] = make_float4(n.x, n.y, n.z, float(valid));
    }
}

// The upstream sliding-window operation transposes the validity window and
// propagates a NaN angle before the final max-pool ignores such centers.
extern "C" __global__ void angle_edge(const float4 *normals, bool *edges, int height, int width, int64_t pixels,
                                      float threshold, float cosine, int packed) {
    for (int64_t pos = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; pos < pixels;
         pos += int64_t(gridDim.x) * blockDim.x) {
        int x = pos % width, y = (pos / width) % height;
        const float4 *frame = normals + pos - int64_t(y) * width - x;
        float4 n = normals[pos];
        bool edge = false, invalid = false;
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int yy = max(0, min(height - 1, y + dy)), xx = max(0, min(width - 1, x + dx));
                int my = max(0, min(height - 1, y + dx)), mx = max(0, min(width - 1, x + dy));
                if (frame[int64_t(my) * width + mx].w == 0.f)
                    continue;
                float4 adjacent = frame[int64_t(yy) * width + xx];
                float dot = reduce3(__fmul_rn(n.x, adjacent.x), __fmul_rn(n.y, adjacent.y),
                                    __fmul_rn(n.z, adjacent.z), packed);
                invalid |= !(dot >= -1.f && dot <= 1.f);
                // acos is decreasing. Use the original comparison in the few
                // ulps around the threshold where rounding could decide it.
                edge |= fabsf(dot - cosine) <= 4 * FLT_EPSILON ? acosf(dot) > threshold : dot < cosine;
            }
        }
        edges[pos] = edge && !invalid;
    }
}

extern "C" __global__ void depth_edges(const float *depth, const bool *mask, const bool *angles, bool *out,
                                       int height, int width, int64_t pixels, int64_t sn, int64_t sy, int64_t sx,
                                       float threshold) {
    for (int64_t pos = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; pos < pixels;
         pos += int64_t(gridDim.x) * blockDim.x) {
        int x = pos % width, y = (pos / width) % height;
        const float *d = depth + (pos / (int64_t(height) * width)) * sn;
        int64_t base = pos - int64_t(y) * width - x;
        float lo = INFINITY, hi = -INFINITY;
        bool edge = false;
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int yy = y + dy, xx = x + dx;
                if (unsigned(yy) >= unsigned(height) || unsigned(xx) >= unsigned(width))
                    continue;
                int64_t i = base + int64_t(yy) * width + xx;
                edge |= angles[i];
                if (mask[i]) {
                    float v = d[yy * sy + xx * sx];
                    if (isnan(v)) {
                        lo = v;
                        hi = v;
                    } else if (!isnan(lo)) {
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                }
            }
        }
        out[pos] = !(((hi - lo) / d[y * sy + x * sx] > threshold) && edge);
    }
}
