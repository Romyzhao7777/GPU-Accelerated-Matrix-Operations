// Benchmark entry point. Runs the CPU baseline and both CUDA kernels repeatedly,
// reports median kernel times, and checks both GPU results against the CPU.
//
// Usage: matmul_bench [N1 N2 ...]  (default sizes: 256 512 1024 2048)

#include "matrix.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

constexpr int GPU_WARMUP_RUNS = 2;
constexpr int GPU_MEASURED_RUNS = 10;

int cpu_runs_for_size(int N) {
    if (N <= 1024) {
        return 5;
    }
    if (N <= 2048) {
        return 3;
    }
    // A single 4096 x 4096 CPU multiplication takes tens of seconds on the
    // benchmark machine. One run keeps the full suite practical.
    return 1;
}

}  // namespace

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
        sizes = {256, 512, 1024, 2048};
    }

    std::printf("Times are medians. GPU: %d warm-up + %d measured runs. "
                "CPU: 5 runs through N=1024, 3 through N=2048, then 1.\n",
                GPU_WARMUP_RUNS, GPU_MEASURED_RUNS);
    std::printf("%8s  %10s  %12s  %12s  %11s  %12s  %8s\n",
                "N", "CPU med", "Naive med", "Tiled med",
                "CPU/Tiled", "Naive/Tiled", "Check");

    bool all_ok = true;
    for (const int N : sizes) {
        if (!cuda_has_enough_memory(N)) {
            std::printf("%8d  %s\n", N, "SKIPPED (insufficient GPU memory)");
            continue;
        }

        try {
            const std::size_t count = static_cast<std::size_t>(N) * N;
            std::vector<float> A(count), B(count), C_cpu(count);
            std::vector<float> C_naive(count), C_tiled(count);
            fill_random(A, 1);
            fill_random(B, 2);

            std::vector<double> cpu_timings_ms;
            const int cpu_runs = cpu_runs_for_size(N);
            cpu_timings_ms.reserve(cpu_runs);
            for (int run = 0; run < cpu_runs; ++run) {
                const auto start = std::chrono::steady_clock::now();
                matmul_cpu(A.data(), B.data(), C_cpu.data(), N);
                const auto stop = std::chrono::steady_clock::now();
                cpu_timings_ms.push_back(
                    std::chrono::duration<double, std::milli>(stop - start).count());
            }
            const TimingStats cpu_stats = summarize_timings(cpu_timings_ms);

            const TimingStats naive_stats =
                matmul_gpu_naive(A.data(), B.data(), C_naive.data(), N,
                                 GPU_WARMUP_RUNS, GPU_MEASURED_RUNS);
            const TimingStats tiled_stats =
                matmul_gpu_tiled(A.data(), B.data(), C_tiled.data(), N,
                                 GPU_WARMUP_RUNS, GPU_MEASURED_RUNS);

            const CompareResult naive_result =
                compare_matrices(C_cpu.data(), C_naive.data(), N);
            const CompareResult tiled_result =
                compare_matrices(C_cpu.data(), C_tiled.data(), N);
            const bool passed = naive_result.passed && tiled_result.passed;
            all_ok = all_ok && passed;

            std::printf("%8d  %10.2f  %12.3f  %12.3f  %10.1fx  %11.2fx  %8s\n",
                        N, cpu_stats.median_ms, naive_stats.median_ms,
                        tiled_stats.median_ms,
                        cpu_stats.median_ms / tiled_stats.median_ms,
                        naive_stats.median_ms / tiled_stats.median_ms,
                        passed ? "OK" : "MISMATCH");
            if (!naive_result.passed) {
                std::printf(
                    "          Naive: %d mismatches, first at %d, max abs error %.3e\n",
                    naive_result.mismatches, naive_result.first_bad_index,
                    naive_result.max_abs_error);
            }
            if (!tiled_result.passed) {
                std::printf(
                    "          Tiled: %d mismatches, first at %d, max abs error %.3e\n",
                    tiled_result.mismatches, tiled_result.first_bad_index,
                    tiled_result.max_abs_error);
            }
        } catch (const std::bad_alloc&) {
            std::printf("%8d  %s\n", N, "SKIPPED (insufficient host memory)");
        }
    }
    return all_ok ? 0 : 1;
}
