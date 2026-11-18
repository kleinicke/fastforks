#include <cuda_fp16.h>
#include <stdint.h>

__device__ float pixel(const half* f, int x, int y, int w, int h, int64_t sy, int64_t sx) {
    if (x < 0 || y < 0 || x >= w || y >= h) return 0.f;
    return __half2float(f[int64_t(y)*sy + int64_t(x)*sx]);
}

// Pinned upstream maps coordinates with (size-1), then samples them with
// align_corners=false. Keep both transforms and zero padding at all four edges.
extern "C" __global__ void selective(const half* left, const half* right, const float* disp, half* out,
        int groups, int channels, int d, int h, int w,
        int64_t lb, int64_t lc, int64_t ly, int64_t lx, int64_t rb, int64_t rc, int64_t ry, int64_t rx, int64_t count) {
    int lane = threadIdx.x % 32;
    for (int64_t i = (int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32; i < count; i += int64_t(gridDim.x)*blockDim.x/32) {
        int x=i%w, y=(i/w)%h, z=(i/w/h)%d, g=(i/w/h/d)%groups;
        int64_t b=i/w/h/d/groups;
        float disparity=disp[((b*d+z)*h+y)*w+x];
        float sum=0.f;
        if (isfinite(disparity) && x >= disparity) {
            float nx = __fmul_rn(x-disparity, 2.f/(w-1))-1.f, ny = __fmul_rn(float(y), 2.f/(h-1))-1.f;
            float px = ((nx+1.f)*w-1.f)*.5f, py = ((ny+1.f)*h-1.f)*.5f;
            if (px > -1.f && px < w && py > -1.f && py < h) {
                int x0=int(floorf(px)), y0=int(floorf(py)), x1=x0+1, y1=y0+1;
                float nw=(x1-px)*(y1-py), ne=(px-x0)*(y1-py), sw=(x1-px)*(py-y0), se=(px-x0)*(py-y0);
                int n=channels/groups;
                for (int j=lane;j<n;j+=32) {
                    int c=g*n+j;
                    const half* f=right+b*rb+c*rc;
                    float value=pixel(f,x0,y0,w,h,ry,rx)*nw;
                    value=fmaf(pixel(f,x1,y0,w,h,ry,rx),ne,value);
                    value=fmaf(pixel(f,x0,y1,w,h,ry,rx),sw,value);
                    value=fmaf(pixel(f,x1,y1,w,h,ry,rx),se,value);
                    sum += __fmul_rn(value,__half2float(left[b*lb+c*lc+y*ly+x*lx]));
                }
            }
        }
        for (int delta=16;delta;delta/=2) sum += __shfl_down_sync(0xffffffff,sum,delta);
        if (!lane) out[i]=__float2half(sum/(channels/groups));
    }
}
