#include <cuda_fp16.h>
#include <math.h>

// Copy the activated half of each packed [gate | up] row into a dense buffer
// so the down projection can read contiguous rows. Round SiLU before multiplying.
extern "C" __global__ void swiglu(const half2* packed, half2* out, int width_pairs, long long count) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        long long pos = (i / width_pairs) * (2 * width_pairs) + i % width_pairs;
        float2 g = __half22float2(packed[pos]);
        half2 activated = __floats2half2_rn(g.x / (1.f + expf(-g.x)), g.y / (1.f + expf(-g.y)));
        out[i] = __hmul2(activated, packed[pos + width_pairs]);
    }
}
