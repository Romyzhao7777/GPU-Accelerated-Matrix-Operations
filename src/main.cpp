// Entry point. Runs the CPU baseline and the naive CUDA kernel for a few matrix
// sizes, checks the GPU result against the CPU result, and prints timings.
// A fuller benchmark harness (repeated runs, all kernels) comes in Stage 4.
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

    std::printf("%8s  %12s  %14s  %10s  %8s\n",
                "N", "CPU (ms)", "Naive GPU (ms)", "Speedup", "Check");

    bool all_ok = true;
    for (const int N : sizes) {
        const std::size_t count = static_cast<std::size_t>(N) * N;
        std::vector<float> A(count), B(count), C_cpu(count), C_gpu(count);
        fill_random(A, 1);
        fill_random(B, 2);

        const auto start = std::chrono::steady_clock::now();
        matmul_cpu(A.data(), B.data(), C_cpu.data(), N);
        const auto stop = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(stop - start).count();

        const float gpu_ms = matmul_gpu_naive(A.data(), B.data(), C_gpu.data(), N);

        const CompareResult r = compare_matrices(C_cpu.data(), C_gpu.data(), N);
        all_ok = all_ok && r.passed;

        std::printf("%8d  %12.2f  %14.3f  %9.1fx  %8s\n",
                    N, cpu_ms, gpu_ms, cpu_ms / gpu_ms, r.passed ? "OK" : "MISMATCH");
        if (!r.passed) {
            std::printf("          %d mismatches, first at index %d, max abs error %.3e\n",
                        r.mismatches, r.first_bad_index, r.max_abs_error);
        }
    }
    return all_ok ? 0 : 1;
}
