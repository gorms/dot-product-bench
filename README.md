# Dot Product Benchmark

**Author:** Michael Gorman  
**Date:** 2026-05-27  
**Hardware:** AMD Ryzen 7 6800HS

Benchmark four dot product implementations across vector sizes chosen to stress L1, L2, L3, and RAM. Each test runs 20 iterations with 3 warm up iterations discarded.

## Implementations

| Name | Description |
|---|---|
| DotSimple | Naive scalar loop |
| DotInner | `std::inner_product` |
| DotEigen | Eigen library (`Eigen::Map` + `.dot()`) |
| DotAVX2 | Hand written AVX2 using 256-bit fused multiply add |
| DotAVX512 | Hand written AVX512 using 512-bit fused multiply add |

AVX512 wasn't tested. AMD Zen3+ doesn't support the instructions.

## Results (mean ms)

| Cache level | Elements | DotSimple | DotInner | DotEigen | DotAVX2 |
|---|---|---|---|---|---|
| L1 (32 KB) | 4,096 | 0.002775 | 0.002770 | 0.000197 | 0.000484 |
| L2 (512 KB) | 65,536 | 0.04696 | 0.04711 | 0.005266 | 0.011788 |
| L3 (16 MB) | 2,097,152 | 1.4391 | 1.4085 | 0.3277 | 0.4342 |
| RAM | 100,000,000 | 66.806 | 66.917 | 26.636 | 26.485 |

## Relative performance (1.00 = fastest at that cache level)

| Cache level | DotSimple | DotInner | DotEigen | DotAVX2 |
|---|---|---|---|---|
| L1 (32 KB) | 14.09x | 14.06x | 1.00x | 2.46x |
| L2 (512 KB) | 8.92x | 8.95x | 1.00x | 2.24x |
| L3 (16 MB) | 4.39x | 4.30x | 1.00x | 1.32x |
| RAM | 2.52x | 2.53x | 1.01x | 1.00x |

## Observations

DotSimple and DotInner are near identical. At each cache level the two results are within noise of each other.

Eigen is the fastest SIMD implementation at L1 and L2. At 4K elements Eigen is ~2.5x faster than the hand written AVX2 implementation.

AVX2 and Eigen converge at RAM. At 100M elements both take ~26.5 ms, roughly 2.5x faster than the scalar implementations. Assume this is now hitting a RAM bottleneck. SIMD is gaining more from cache than scalar is, so it also loses more when data falls out of it.

## Build

```sh
make
./bench
```

Requires `g++` with AVX2 support and Eigen3 (`sudo pacman -S eigen` on Arch Linux).
