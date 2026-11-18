#include <cuda_fp16.h>
#include <stdint.h>

// NHWC input, kernel-major weights [9, 2C]. Neighboring lanes load neighboring
// channels. Each output thread accumulates one channel from each gate operand.
extern "C" __global__ void depthwise_gate(const half* input, const half* weight, const half* bias,
        half* output, int h, int w, int c, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int ch = i % c;
        const int64_t pixel = i / c;
        const int x = pixel % w, y = pixel / w % h;
        float a = 0.f, b = 0.f;
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (unsigned(y + dy) < unsigned(h) && unsigned(x + dx) < unsigned(w)) {
                    const int64_t p = (pixel + int64_t(dy) * w + dx) * (2 * c) + ch;
                    const int k = ((dy + 1) * 3 + dx + 1) * (2 * c) + ch;
                    a = fmaf(__half2float(input[p]), __half2float(weight[k]), a);
                    b = fmaf(__half2float(input[p + c]), __half2float(weight[k + c]), b);
                }
            }
        }
        const half left = __float2half_rn(a + __half2float(bias[ch]));
        const half right = __float2half_rn(b + __half2float(bias[ch + c]));
        output[i] = __hmul(left, right);
    }
}
