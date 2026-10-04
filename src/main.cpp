// Entry point. Runs the CPU baseline and both CUDA kernels for a few matrix
// sizes, checks both GPU results against the CPU result, and prints timings.
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

    std::printf("%8s  %10s  %12s  %12s  %11s  %12s  %8s\n",
                "N", "CPU (ms)", "Naive (ms)", "Tiled (ms)",
                "CPU/Tiled", "Naive/Tiled", "Check");

    bool all_ok = true;
    for (const int N : sizes) {
        const std::size_t count = static_cast<std::size_t>(N) * N;
        std::vector<float> A(count), B(count), C_cpu(count);
        std::vector<float> C_naive(count), C_tiled(count);
        fill_random(A, 1);
        fill_random(B, 2);

        const auto start = std::chrono::steady_clock::now();
        matmul_cpu(A.data(), B.data(), C_cpu.data(), N);
        const auto stop = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(stop - start).count();

        const float naive_ms = matmul_gpu_naive(A.data(), B.data(), C_naive.data(), N);
        const float tiled_ms = matmul_gpu_tiled(A.data(), B.data(), C_tiled.data(), N);

        const CompareResult naive_result =
            compare_matrices(C_cpu.data(), C_naive.data(), N);
        const CompareResult tiled_result =
            compare_matrices(C_cpu.data(), C_tiled.data(), N);
        const bool passed = naive_result.passed && tiled_result.passed;
        all_ok = all_ok && passed;

        std::printf("%8d  %10.2f  %12.3f  %12.3f  %10.1fx  %11.2fx  %8s\n",
                    N, cpu_ms, naive_ms, tiled_ms, cpu_ms / tiled_ms,
                    naive_ms / tiled_ms, passed ? "OK" : "MISMATCH");
        if (!naive_result.passed) {
            std::printf("          Naive: %d mismatches, first at %d, max abs error %.3e\n",
                        naive_result.mismatches, naive_result.first_bad_index,
                        naive_result.max_abs_error);
        }
        if (!tiled_result.passed) {
            std::printf("          Tiled: %d mismatches, first at %d, max abs error %.3e\n",
                        tiled_result.mismatches, tiled_result.first_bad_index,
                        tiled_result.max_abs_error);
        }
    }
    return all_ok ? 0 : 1;
}
