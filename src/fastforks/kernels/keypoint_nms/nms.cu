#include <stdint.h>

// Threshold first: most pixels never need to read their neighbours.
extern "C" __global__ void maxima(const float* image, bool* keep, int height, int width,
                                  int radius, float threshold, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const float value = image[i];
        bool peak = value > threshold;
        if (peak) {
            const int x = i % width, y = (i / width) % height;
            const float* plane = image + i - int64_t(y) * width - x;
            for (int row = max(0, y-radius); peak && row <= min(height-1, y+radius); ++row)
                for (int col = max(0, x-radius); col <= min(width-1, x+radius); ++col)
                    if (!(value >= plane[int64_t(row)*width+col])) { peak = false; break; }
        }
        keep[i] = peak;
    }
}
