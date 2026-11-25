#include <stdint.h>

extern "C" __global__ void bgr_bytes(const float* src, uint8_t* dst, int64_t plane, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += int64_t(blockDim.x)*gridDim.x) {
        const int channel = (i / plane) % 3;
        dst[i] = uint8_t(fminf(255.f, fmaxf(0.f, nearbyintf(src[i + (2 - 2*channel)*plane] * 255.f))));
    }
}
