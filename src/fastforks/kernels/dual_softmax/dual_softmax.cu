#include <stdint.h>
#include <math.h>

// One block reads a 32-column x 128-row tile. Accumulate the two denominators
// together, visiting each logit once and writing just the partial sums.
extern "C" __global__ void denominators(const float* logits, const float* row_max, const float* col_max,
        float* row_parts, float* col_parts, int rows, int cols) {
    __shared__ float partial[8][32];
    int lane = threadIdx.x, warp = threadIdx.y;
    int col = blockIdx.x * 32 + lane, start = blockIdx.y * 128;
    float col_sum = 0.f, cm = col < cols ? col_max[col] : 0.f;
    for (int row = start + warp; row < rows && row < start + 128; row += 8) {
        float row_sum = 0.f;
        if (col < cols) {
            float x = logits[int64_t(row) * cols + col];
            row_sum = expf(x - row_max[row]);
            col_sum += expf(x - cm);
        }
        for (int s = 16; s; s /= 2) row_sum += __shfl_down_sync(0xffffffff, row_sum, s);
        if (!lane) row_parts[int64_t(row) * gridDim.x + blockIdx.x] = row_sum;
    }
    partial[warp][lane] = col_sum;
    __syncthreads();
    if (!warp && col < cols) {
        float sum = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum += partial[w][lane];
        col_parts[int64_t(blockIdx.y) * cols + col] = sum;
    }
}

extern "C" __global__ void probabilities(float* logits, const float* row_max, const float* col_max,
        const float* row_sum, const float* col_sum, int cols, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x) {
        int row = i / cols, col = i % cols;
        float x = logits[i];
        logits[i] = (expf(x - row_max[row]) / row_sum[row]) * (expf(x - col_max[col]) / col_sum[col]);
    }
}

// Each warp owns a row. Ballots retain all tied maxima in ascending column
// order. The counting pass provides offsets for an exactly sized result.
template<bool emit>
__device__ void select_row(const float* p, const float* row_max, const float* col_max,
        int64_t* counts, const int64_t* ends, int64_t* pairs, float* scores,
        float threshold, int rows, int cols) {
    int lane = threadIdx.x, row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    float peak = row_max[row];
    int total = 0;
    if (peak > threshold) {
        int64_t begin = emit && row ? ends[row - 1] : 0;
        for (int base = 0; base < cols; base += 32) {
            int col = base + lane;
            bool keep = col < cols && col_max[col] == peak && p[int64_t(row) * cols + col] == peak;
            unsigned votes = __ballot_sync(0xffffffff, keep);
            if (emit && keep) {
                int64_t i = begin + total + __popc(votes & ((1u << lane) - 1));
                pairs[2*i] = row;
                pairs[2*i + 1] = col;
                scores[i] = peak;
            }
            total += __popc(votes);
        }
    }
    if (!emit && !lane) counts[row] = total;
}

extern "C" __global__ void count_matches(const float* p, const float* row_max, const float* col_max,
        int64_t* counts, float threshold, int rows, int cols) {
    select_row<false>(p,row_max,col_max,counts,nullptr,nullptr,nullptr,threshold,rows,cols);
}
extern "C" __global__ void write_matches(const float* p, const float* row_max, const float* col_max,
        const int64_t* ends, int64_t* pairs, float* scores, float threshold, int rows, int cols) {
    select_row<true>(p,row_max,col_max,nullptr,ends,pairs,scores,threshold,rows,cols);
}
