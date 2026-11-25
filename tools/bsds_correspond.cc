// C ABI for the matcher in an extracted UC Berkeley BSDS500 benchmark.
#include <cmath>

#include "Matrix.hh"
#include "Random.hh"
#include "match.hh"

extern "C" int bsds_correspond(
    int height, int width, double* predicted, double* annotated,
    unsigned char* predicted_match, unsigned char* annotated_match)
{
    const double tolerance = 0.0075 * std::sqrt(double(height) * height + double(width) * width);
    try {
        // The benchmark samples outlier edges. Reset each call independently.
        Random::rand.reseed(1);
        Matrix a, b;
        matchEdgeMaps(Matrix(height, width, predicted), Matrix(height, width, annotated),
                      tolerance, 100 * tolerance, a, b);
        for (int index = 0; index < height * width; ++index) {
            predicted_match[index] = a(index) > 0;
            annotated_match[index] = b(index) > 0;
        }
        return 0;
    } catch (...) {
        return 1;
    }
}
