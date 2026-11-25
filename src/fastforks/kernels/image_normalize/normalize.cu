#include <stdint.h>

extern "C" __global__ void normalize(const float* input, const float* mean, const float* stddev,
                                     float* out, int64_t plane, int channels, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += int64_t(blockDim.x)*gridDim.x) {
        const int channel = (i / plane) % channels;
        out[i] = (input[i] - mean[channel]) / stddev[channel];
    }
}
