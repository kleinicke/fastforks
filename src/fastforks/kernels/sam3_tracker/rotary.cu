#include <cuda_fp16.h>
#include <stdint.h>

// Each row is one spatial token, 128 adjacent even/odd channel pairs.
// The phase repeats for each memory frame; object-pointer rows stay untouched.
// Two rounded fp32 products and an fp32 sum match the eager torch expression.
extern "C" __global__ void rotate_pairs(half2* keys, const float2* cosine, const float2* sine,
        int spatial, int tokens, int period, int pairs, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(blockDim.x) * gridDim.x) {
        int64_t row = i / pairs;
        int channel = i % pairs;
        int t = row % spatial;
        int64_t offset = (row / spatial * tokens + t) * pairs + channel;
        int phase = (t % period) * pairs + channel;
        float2 x = __half22float2(keys[offset]);
        float2 c = cosine[phase], s = sine[phase];
        float a = __fsub_rn(__fmul_rn(x.x,c.x), __fmul_rn(x.y,s.x));
        float b = __fadd_rn(__fmul_rn(x.y,c.y), __fmul_rn(x.x,s.y));
        keys[offset] = __floats2half2_rn(a,b);
    }
}
