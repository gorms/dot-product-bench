#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <chrono>
#include <numeric>
#include <functional>
#include <immintrin.h>
#include "dot_avx512.hpp"

float dotSimple(std::vector<float>& x, std::vector<float>& y) {
    // Make vectors match in size
    assert(x.size() == y.size());

    float sum = 0;

    for (size_t i = 0; i < x.size(); i++) {
        sum += x[i] * y[i];
    }

    return sum;
}

float dotInner(std::vector<float>& x, std::vector<float>& y) {
    // Make vectors match in size
    assert(x.size() == y.size());
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);
}

float dotAVX2(std::vector<float>& x, std::vector<float>& y) {
    assert(x.size() == y.size());

    size_t n = x.size();
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();

    // Iterate over elements in the vectors, 8 at a time.
    for (; i + 8 <= n; i += 8) {
        // Load 8 floats into each of the two registers.
        // Load is unaligned. We could improve this?
        __m256 a = _mm256_loadu_ps(&x[i]);
        __m256 b = _mm256_loadu_ps(&y[i]);
        // Fused multiply add.
        // Sum increases separately in each lane
        sum = _mm256_fmadd_ps(a, b, sum);
    }

    // horizontal sum of the 8 lanes
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    // vertical add the top 4 and bottom 4 floats
    __m128 s = _mm_add_ps(lo, hi);
    // horizontal add 0,1 and 2,3. results get placed in lanes 0,1,2,3
    s = _mm_hadd_ps(s, s);
    // horizontal add 0,1 and 2,3. result gets placed in lane 0
    s = _mm_hadd_ps(s, s);
    // Extract lane 0 float
    float result = _mm_cvtss_f32(s);

    // scalar tail for elements that don't fill a full 8-wide chunk
    for (; i < n; i++)
        result += x[i] * y[i];

    return result;
}


void testAlgo(std::function<float(std::vector<float>&, std::vector<float>&)> fn,
              std::vector<float>& x,
              std::vector<float>& y,
              std::string s) {
    auto start = std::chrono::high_resolution_clock::now();
    fn(x, y);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << s << ": \t" << duration.count() << " ms\n";
}

void testAll(size_t n) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(n);
    std::vector<float> y(n);

    for (size_t i = 0; i < n; i++) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    testAlgo(dotSimple, x, y, "DotSimple");
    testAlgo(dotInner, x, y, "DotInner");
    testAlgo(dotAVX2, x, y, "DotAVX2");
    if (__builtin_cpu_supports("avx512f")) {
        testAlgo(dotAVX512, x, y, "DotAVX512");
    }
}

int main() {
    std::cout << "Round 2\n";

    testAll(1e8);
    return 0;
}
