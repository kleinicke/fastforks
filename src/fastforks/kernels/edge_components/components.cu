#include <stdint.h>

// A foreground pixel survives if its four-connected component contains at
// least twelve pixels. Stop the breadth-first search as soon as that is known.
extern "C" __global__ void keep_twelve(const bool* mask, bool* out, int height, int width, int64_t count) {
    const int plane = height * width;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < count; i += (int64_t)blockDim.x * gridDim.x) {
        out[i] = false;
        if (!mask[i]) continue;
        int queue[12];
        int size = 1;
        queue[0] = i % plane;
        const bool* image = mask + (i / plane) * plane;
        for (int head = 0; head < size && size < 12; ++head) {
            const int p = queue[head], x = p % width;
            const int next[4] = {x ? p-1 : -1, x+1 < width ? p+1 : -1, p-width, p+width};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = next[j];
                if (n < 0 || n >= plane || !image[n]) continue;
                bool seen = false;
                for (int k = 0; k < size; ++k) seen |= queue[k] == n;
                if (!seen) {
                    queue[size++] = n;
                    if (size == 12) break;
                }
            }
        }
        out[i] = size == 12;
    }
}
