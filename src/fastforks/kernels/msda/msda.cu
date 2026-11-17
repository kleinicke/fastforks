#include <cuda_fp16.h>
#include <stdint.h>

// One lane per value channel. All channels of a head share offsets and attention weights.
extern "C" __global__ void deform_sample(const half* value, const int64_t* shapes, const int64_t* start,
    const half* locations, const half* weights, half* out, int length, int queries, int heads,
    int channels, int levels, int points, int64_t count) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t row = i / channels;
        int head = row % heads, batch = row / heads / queries, channel = i % channels;
        half total = __float2half(0.f);
        for (int l = 0; l < levels; ++l) {
            int h = shapes[2*l], w = shapes[2*l+1];
            float accum = 0.f;
            for (int p = 0; p < points; ++p) {
                int64_t k = (row*levels+l)*points+p;
                half2 uv = reinterpret_cast<const half2*>(locations)[k];
                half2 grid = __hsub2(__hmul2(uv, __float2half2_rn(2.f)), __float2half2_rn(1.f));
                float x = ((__low2float(grid)+1.f)*w-1.f)*0.5f, y = ((__high2float(grid)+1.f)*h-1.f)*0.5f;
                if (!(x>-1.f && x<w && y>-1.f && y<h)) continue;
                int left = __float2int_rd(x), top = __float2int_rd(y);
                float dx=x-left, dy=y-top, sample=0.f;
                #pragma unroll
                for (int iy=0; iy<2; ++iy) {
                    #pragma unroll
                    for (int ix=0; ix<2; ++ix) {
                        int yy=top+iy, xx=left+ix;
                        if (xx>=0 && xx<w && yy>=0 && yy<h) {
                            int64_t token = batch*(int64_t)length + start[l] + yy*w + xx;
                            sample += (ix?dx:1.f-dx)*(iy?dy:1.f-dy)*__half2float(value[(token*heads+head)*channels+channel]);
                        }
                    }
                }
                accum += __half2float(__hmul(__float2half_rn(sample),weights[k]));
            }
            total = __hadd(total,__float2half_rn(accum));
        }
        out[i]=total;
    }
}
