#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <chrono>

float dot(std::vector<float> x, std::vector<float> y) {
    /* Make vectors match in size */
    assert(x.size() == y.size());

    float z = 0;

    for (size_t i = 0; i < x.size(); i++) {
        z += x[i] * y[i];
    }

    return z;
}

auto test(u_int64_t n) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(n);
    std::vector<float> y(n);

    for (size_t i = 0; i < n; i++)
    {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    auto start = std::chrono::high_resolution_clock::now();
    dot(x, y);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << duration.count() << " ms\n";
}

int main() {
    std::cout << "Round 2\n";

    test(10000);
    return 0;
}
