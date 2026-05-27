/**
 * @author Michael Gorman
 * @date   2026-05-27
 * @brief  AVX-512 dot product implementation. Compiled separately with
 *         -mavx512f to isolate AVX-512 instructions from the main translation unit.
 */

#include "dot_avx512.hpp"

#include <cassert>
#include <immintrin.h>

float dotAVX512(std::vector<float>& x, std::vector<float>& y) {
    assert(x.size() == y.size());

    size_t n = x.size();
    size_t i = 0;
    // Zero the sum
    __m512 sum = _mm512_setzero_ps();

    for (; i + 16 <= n; i += 16) {
        // Load 16 floats from each vector into the 512 bit registers
        __m512 a = _mm512_loadu_ps(&x[i]);
        __m512 b = _mm512_loadu_ps(&y[i]);
        // Vertically add them
        sum = _mm512_fmadd_ps(a, b, sum);
    }

    // Horizontally add the 16 floats.
    // A lot simpler than AVX2
    float result = _mm512_reduce_add_ps(sum);

    // scalar tail
    for (; i < n; i++)
        result += x[i] * y[i];

    return result;
}
