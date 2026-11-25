#include <cuda_fp16.h>
#include <stdint.h>

// Consecutive lanes read one head. The quarter swap inside each 32-wide
// axis is an XOR by 16. All head and rotation vectors are in fp16; preserve
// the intermediate roundings of PyTorch mul, addcmul and addcmul_.
extern "C" __global__ void rotary(const half* norm, const half* weight, const half* bias,
        const half* cosine, const half* sine, half* out, int heads, int64_t count) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (int64_t)gridDim.x * blockDim.x) {
        int64_t qk = i / (64 * heads);
        int channel = int(i & 63);
        int affine = (int(qk & 1) * 64) + channel;
        int64_t pos = (qk / 2) * 64 + channel;
        half c = cosine[pos], s = sine[pos];
        half b = __float2half_rn(fmaf(__half2float(bias[affine ^ 16]), __half2float(s),
                                      __half2float(__hmul_rn(bias[affine], c))));
        half a = __float2half_rn(fmaf(__half2float(norm[i]), __half2float(__hmul_rn(weight[affine], c)),
                                      __half2float(b)));
        out[i] = __float2half_rn(fmaf(__half2float(norm[i ^ 16]), __half2float(__hmul_rn(weight[affine ^ 16], s)),
                                     __half2float(a)));
    }
}
