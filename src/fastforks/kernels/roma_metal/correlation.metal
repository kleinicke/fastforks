// One SIMD group computes a single offset of the zero-padded bilinear correlation window.
#include <metal_stdlib>
using namespace metal;

kernel void correlate_float(device const float* a, device const float* b, device const float* warp,
                            device float* out, constant uint& H, constant uint& W, constant uint& C,
                            constant int& R, uint gid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint index = gid / 32;
    uint side = 2 * R + 1;
    uint pixel = index / (side*side), offset = index % (side*side), batch = pixel / (W*H);
    float u = warp[2*pixel] + 2.0f * (int(offset % side) - R) / float(W);
    float v = warp[2*pixel+1] + 2.0f * (int(offset / side) - R) / float(H);
    float x = ((u+1)*W-1)*0.5f, y = ((v+1)*H-1)*0.5f;
    if (x <= -1 || x >= float(W) || y <= -1 || y >= float(H)) {
        if (lane == 0) out[index] = 0;
        return;
    }
    int ix = int(floor(x)), iy = int(floor(y));
    float dx = x-ix, dy = y-iy, sum = 0;
    for (uint c = lane; c < C; c += 32) {
        float sampled = 0;
        for (int j = 0; j <= 1; ++j) {
            for (int i = 0; i <= 1; ++i) {
                int xx = ix+i, yy = iy+j;
                if (xx >= 0 && xx < int(W) && yy >= 0 && yy < int(H))
                    sampled += b[((size_t(batch)*H+yy)*W+xx)*C+c] * (i ? dx : 1-dx) * (j ? dy : 1-dy);
            }
        }
        sum += a[size_t(pixel)*C+c] * sampled;
    }
    sum = simd_sum(sum);
    if (lane == 0) out[index] = sum * rsqrt(float(C));
}
