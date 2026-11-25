#include <cuda_fp16.h>
#include <stdint.h>

// grid_sample and the elementwise product preserve the feature dtype under autocast.
// The subsequent torch.sum promotes its accumulator and result to fp32.
__device__ float product(half ref, float sampled) {
    return __half2float(__hmul_rn(ref, __float2half_rn(sampled)));
}
__device__ float product(float ref, float sampled) {
    return ref * sampled;
}

// A warp samples one projected pixel. Neighbour views are selected by the
// model before this operation (the order can differ between reference views).
template <typename T>
__device__ void correlate(const T* ref, const T* target, const half2* grid,
        float* volume, int channels, int height, int width, int depths, int neighbor, int neighbors, int64_t count) {
    const int lane = threadIdx.x % 32;
    const int64_t pixels = int64_t(height) * width;
    for (int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         i < count; i += int64_t(gridDim.x) * (blockDim.x / 32)) {
        const int64_t batch = i / (depths * pixels);
        const float2 uv = __half22float2(grid[i]);
        const float x = ((uv.x + 1.f) * .5f) * (width - 1);
        const float y = ((uv.y + 1.f) * .5f) * (height - 1);
        float dot = 0;
        if (x > -1.f && x < width && y > -1.f && y < height) {
            const int x0 = __float2int_rd(x), y0 = __float2int_rd(y);
            const T* left = ref + (batch * pixels + i % pixels) * channels;
            const T* right = target + batch * pixels * channels;
            for (int c = lane; c < channels; c += 32) {
                float sample = 0;
                #pragma unroll
                for (int dy = 0; dy < 2; ++dy) {
                    #pragma unroll
                    for (int dx = 0; dx < 2; ++dx) {
                        int xx = x0 + dx, yy = y0 + dy;
                        if (unsigned(xx) < unsigned(width) && unsigned(yy) < unsigned(height)) {
                            float wx = dx ? x - x0 : x0 + 1 - x;
                            float wy = dy ? y - y0 : y0 + 1 - y;
                            sample = fmaf(wx * wy, float(right[(int64_t(yy) * width + xx) * channels + c]), sample);
                        }
                    }
                }
                dot += product(left[c], sample);
            }
        }
        for (int s = 16; s; s /= 2) dot += __shfl_down_sync(0xffffffffu, dot, s);
        if (!lane) {
            float cost = dot / sqrtf(float(channels));
            if (neighbor) cost += volume[i];
            volume[i] = neighbor + 1 == neighbors ? cost / neighbors : cost;
        }
    }
}

extern "C" __global__ void plane_float(const float* ref, const float* target, const half2* grid, float* volume,
        int c, int h, int w, int d, int neighbor, int neighbors, int64_t count) {
    correlate(ref, target, grid, volume, c, h, w, d, neighbor, neighbors, count);
}
extern "C" __global__ void plane_half(const half* ref, const half* target, const half2* grid, float* volume,
        int c, int h, int w, int d, int neighbor, int neighbors, int64_t count) {
    correlate(ref, target, grid, volume, c, h, w, d, neighbor, neighbors, count);
}
