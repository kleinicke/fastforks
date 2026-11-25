#include <cuda_fp16.h>
#include <stdint.h>

// Only the outer two rows/columns of the zero-correlation bias depend on
// position. The rest of the map uses the center of the same 5x5 tile.
extern "C" __global__ void pack(const half* motion, const half* tile, const half* flow, half* out,
        int height, int width, int64_t sb, int64_t sc, int64_t sy, int64_t sx, int64_t count) {
    for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<count; i+=int64_t(gridDim.x)*blockDim.x) {
        int c=i%128;
        int64_t p=i/128;
        int x=p%width, y=(p/width)%height;
        if (c < 126) {
            int tx=x<2 ? x : x>=width-2 ? 5-width+x : 2;
            int ty=y<2 ? y : y>=height-2 ? 5-height+y : 2;
            half sum=__hadd(motion[p*126+c],tile[(ty*5+tx)*126+c]);
            out[i]=__hlt(sum,__float2half(0.f)) ? __float2half(0.f) : sum;
        } else {
            out[i]=flow[(p/width/height)*sb+(c-126)*sc+y*sy+x*sx];
        }
    }
}
