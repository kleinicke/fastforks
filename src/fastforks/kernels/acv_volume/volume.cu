#include <cuda_fp16.h>

// Convert an integer coordinate through the normalized FP32 grid used by align_corners=True.
__device__ float grid_coordinate(int pos, int size) {
    float normalized = __fmul_rn(float(pos), 2.f / (size - 1)) - 1.f;
    return ((normalized + 1.f) * .5f) * (size - 1);
}

__device__ float at(const half* features, int x, int y, int w, int h, int c) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0.f;
    return __half2float(features[(static_cast<long long>(y) * w + x) * c]);
}

extern "C" __global__ void weighted(const half* features, const long long* disparity,
        const float* attention, half* output, int batches, int depth, int h, int w, int channels) {
    // Each warp covers one pixel's 16 left and 16 right feature channels for the released checkpoints.
    int lane = threadIdx.x & 31;
    int x = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (x >= w) return;
    long long stride = static_cast<long long>(h) * w * channels;
    for (int plane = blockIdx.z; plane < batches * depth; plane += gridDim.z) {
        int b = plane / depth;
        for (int y = blockIdx.y; y < h; y += gridDim.y) {
            long long sample = (static_cast<long long>(plane) * h + y) * w + x;
            for (int channel = lane; channel < 2 * channels; channel += 32) {
                float value;
                if (channel < channels) {
                    value = __half2float(features[b * stride + (static_cast<long long>(y) * w + x) * channels + channel]);
                } else {
                    const half* right = features + (batches + b) * stride + channel - channels;
                    float px = grid_coordinate(x - static_cast<int>(disparity[sample]), w);
                    float py = grid_coordinate(y, h);
                    int x0 = __float2int_rd(px), y0 = __float2int_rd(py);
                    float dx = px - x0, dy = py - y0;
                    value = at(right, x0, y0, w, h, channels) * ((1.f - dx) * (1.f - dy));
                    value = fmaf(at(right, x0 + 1, y0, w, h, channels), dx * (1.f - dy), value);
                    value = fmaf(at(right, x0, y0 + 1, w, h, channels), (1.f - dx) * dy, value);
                    value = fmaf(at(right, x0 + 1, y0 + 1, w, h, channels), dx * dy, value);
                }
                output[sample * (2 * channels) + channel] = __float2half(value * attention[sample]);
            }
        }
    }
}
