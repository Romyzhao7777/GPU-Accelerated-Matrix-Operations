// Entry point. In Stage 1 it times the CPU baseline for a few matrix sizes.
// GPU implementations and a fuller benchmark harness are added in later stages.
//
// Usage: matmul_bench [N1 N2 ...]     (default sizes: 256 512 1024)

#include "matrix.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        const int n = std::atoi(argv[i]);
        if (n <= 0) {
            std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
            return 1;
        }
        sizes.push_back(n);
    }
    if (sizes.empty()) {
        sizes = {256, 512, 1024};
    }

    std::printf("%8s  %12s  %10s\n", "N", "CPU (ms)", "GFLOP/s");

    for (const int N : sizes) {
        const std::size_t count = static_cast<std::size_t>(N) * N;
        std::vector<float> A(count), B(count), C(count);
        fill_random(A, 1);
        fill_random(B, 2);

        const auto start = std::chrono::steady_clock::now();
        matmul_cpu(A.data(), B.data(), C.data(), N);
        const auto stop = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        // An N x N matmul does N^3 multiplies and N^3 adds.
        const double gflops = 2.0 * N * N * N / (ms * 1e6);
        std::printf("%8d  %12.2f  %10.2f\n", N, ms, gflops);
    }
    return 0;
}
