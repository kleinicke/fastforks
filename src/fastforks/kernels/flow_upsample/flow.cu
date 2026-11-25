#include <stdint.h>
#include <math.h>

// Merge the nine-neighbour mask softmax, zero-padded unfolding, weighted sum and pixel shuffle.
extern "C" __global__ void convex(const float* flow, const float* mask, float* out,
    int height, int width, int scale, int64_t count) {
    const int out_w = scale * width, out_h = scale * height;
    const int64_t plane = (int64_t)height * width;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)blockDim.x * gridDim.x) {
        int x = i % out_w, y = (i / out_w) % out_h;
        int64_t b = i / out_w / out_h;
        int low_x = x / scale, low_y = y / scale;
        int inner = (y % scale) * scale + x % scale;
        int64_t step = scale * scale * plane;
        const float* logits = mask + b * 9 * step + inner * plane + low_y * width + low_x;
        float w[9], largest = -INFINITY;
        #pragma unroll
        for (int j=0; j<9; ++j) { w[j]=logits[j*step]; largest=fmaxf(largest,w[j]); }
        float sum = 0.f;
        #pragma unroll
        for (int j=0; j<9; ++j) { w[j]=expf(w[j]-largest); sum+=w[j]; }
        float vx=0.f, vy=0.f;
        #pragma unroll
        for (int j=0; j<9; ++j) {
            const int xx=low_x + j%3-1, yy=low_y+j/3-1;
            if (xx>=0 && xx<width && yy>=0 && yy<height) {
                const int64_t k=b*2*plane+yy*width+xx;
                const float p=w[j]/sum;
                vx += p * (scale * flow[k]);
                vy += p * (scale * flow[k + plane]);
            }
        }
        const int64_t out_plane = (int64_t)out_h * out_w;
        out[i+b*out_plane]=vx;
        out[i+(b+1)*out_plane]=vy;
    }
}
