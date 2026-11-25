#include <cuda_fp16.h>
#include <stdint.h>

// The two sigmoid gates have already been evaluated once per channel or pixel.
// Preserve half multiplication and addition rounding while writing the token
// layout required by the next projection. Both feature inputs may be strided.
extern "C" __global__ void interaction(const half* channel_features, const half* spatial_features,
        const half* channels, const half* spatial, half* out, int n, int c,
        int64_t cb, int64_t cn, int64_t cc, int64_t sb, int64_t sn, int64_t sc, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int ch = i % c, pixel = i / c % n;
        int64_t batch = i / c / n;
        half a = channel_features[batch*cb + pixel*cn + ch*cc];
        half b = spatial_features[batch*sb + pixel*sn + ch*sc];
        out[i] = __hadd_rn(__hmul_rn(a, channels[batch*c + ch]), __hmul_rn(b, spatial[batch*n + pixel]));
    }
}
