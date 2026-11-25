#include <cuda_fp16.h>

extern "C" __global__ void affine_silu_fp16(
    uint4* x, const uint4* scale, const uint4* bias, int channel_vectors, long long image_vectors
) {
    x += blockIdx.y * image_vectors;
    scale += blockIdx.y * channel_vectors;
    bias += blockIdx.y * channel_vectors;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < image_vectors;
            i += gridDim.x * (long long)blockDim.x) {
        int c = i % channel_vectors;
        uint4 current = x[i], a = scale[c], b = bias[c];
        auto* values = reinterpret_cast<__half*>(&current);
        const auto* scales = reinterpret_cast<const __half*>(&a);
        const auto* biases = reinterpret_cast<const __half*>(&b);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // Round the affine output to FP16 before the activation, as two separate PyTorch operations do.
            float y = __half2float(__float2half_rn(__fmaf_rn(
                __half2float(values[j]), __half2float(scales[j]), __half2float(biases[j])
            )));
            values[j] = __float2half_rn(__fdiv_rn(y, 1.0f + expf(-y)));
        }
        x[i] = current;
    }
}
