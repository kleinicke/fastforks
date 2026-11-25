#include <stdint.h>

// Filter one image axis, including Pillow's rounding and saturation after each pass.
extern "C" __global__ void resample_u8(const uint8_t* src, const int64_t* indices, const int32_t* weights,
    uint8_t* dst, int height, int width, int output_size, int taps, int vertical, int64_t count) {
    const int out_h = vertical ? output_size : height;
    const int out_w = vertical ? width : output_size;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)blockDim.x * gridDim.x) {
        const int col = i % out_w;
        const int row = (i / out_w) % out_h;
        const int64_t plane = i / out_w / out_h;
        const int64_t coeff = (int64_t)(vertical ? row : col) * taps;
        const int64_t base = plane * height * width + (vertical ? col : row * (int64_t)width);
        const int stride = vertical ? width : 1;
        int acc = 1 << 21;
        for (int j = 0; j < taps; ++j) {
            acc += src[base + indices[coeff + j] * stride] * weights[coeff + j];
        }
        dst[i] = min(255, max(0, acc >> 22));
    }
}
