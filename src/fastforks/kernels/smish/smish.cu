#include <cuda_fp16.h>
#include <stdint.h>

// Preserve the fp16 rounding boundaries of sigmoid, addition, log and tanh.
extern "C" __global__ void smish(const half* input, half* output, int64_t count) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)blockDim.x * gridDim.x) {
        const half x = input[i];
        const half s = __float2half_rn(1.f / (1.f + expf(-__half2float(x))));
        const half logarithm = __float2half_rn(logf(__half2float(__hadd(__float2half(1.f), s))));
        output[i] = __hmul(x, __float2half_rn(tanhf(__half2float(logarithm))));
    }
}
