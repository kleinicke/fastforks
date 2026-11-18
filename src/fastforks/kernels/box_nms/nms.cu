#include <math.h>

__device__ bool overlap(float4 a, float4 b, float threshold, float eps) {
    // NaNs in a coordinate cannot suppress a box (torch.minimum/maximum propagate).
    if (isnan(a.x+a.y+a.z+a.w) || isnan(b.x+b.y+b.z+b.w)) return false;
    float w = fmaxf(0.f, fminf(a.z,b.z)-fmaxf(a.x,b.x));
    float h = fmaxf(0.f, fminf(a.w,b.w)-fmaxf(a.y,b.y));
    float area = w*h;
    float sum = __fadd_rn((a.z-a.x)*(a.w-a.y), (b.z-b.x)*(b.w-b.y));
    float den = sum-area;
    return area/(eps > 0.f ? fmaxf(den,eps) : den) > threshold;
}

// Tiny candidate sets fit entirely in one block. Compare each retained box
// with the remaining candidates and preserve the priority order of the rows.
extern "C" __global__ void small(const float4* boxes, bool* keep, int n, float threshold, float eps) {
    __shared__ bool remaining[256];
    int col = threadIdx.x;
    remaining[col] = col < n;
    __syncthreads();
    for (int i=0; i<n; ++i) {
        if (remaining[i] && col > i && col < n && remaining[col])
            remaining[col] = !overlap(boxes[i],boxes[col],threshold,eps);
        __syncthreads();
    }
    if (col < n) keep[col] = remaining[col];
}

// At most 256 rows of the packed overlap matrix exist at any time. Each warp
// writes 32 pairwise decisions; boxes ordered ahead of the row are excluded.
extern "C" __global__ void masks(const float4* boxes, unsigned* mask, int n, int first, int rows,
        int words, float threshold, float eps) {
    int local = blockIdx.y*blockDim.y+threadIdx.y;
    if (local >= rows) return;
    int row = first + local, word = blockIdx.x, col = word*32 + threadIdx.x;
    bool hit = col > row && col < n && overlap(boxes[row], boxes[col], threshold, eps);
    unsigned bits = __ballot_sync(0xffffffff,hit);
    if (!threadIdx.x) mask[local*words+word] = bits;
}

extern "C" __global__ void select_rows(const unsigned* mask, unsigned* gone, bool* keep,
        int first, int rows, int words) {
    __shared__ bool take;
    for (int i=0; i<rows; ++i) {
        int row=first+i;
        if (!threadIdx.x) {
            take = !(gone[row/32] & (1u << (row%32)));
            keep[row] = take;
        }
        __syncthreads();
        if (take)
            for (int word=row/32+threadIdx.x; word<words; word+=blockDim.x)
                gone[word] |= mask[i*words+word];
        __syncthreads();
    }
}
