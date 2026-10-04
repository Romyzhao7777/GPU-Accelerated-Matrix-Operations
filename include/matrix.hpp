#pragma once

#include <vector>

// Matrix layout used throughout the project:
//   - Square N x N matrices of float.
//   - Stored as one contiguous, row-major array: element (row, col) lives at
//     index row * N + col.
// A flat array (instead of vector<vector<float>>) is exactly what cudaMemcpy
// needs later, so CPU and GPU code can share the same buffers.

// C = A * B, computed on the CPU. This is the correctness and performance
// baseline for every GPU implementation.
void matmul_cpu(const float* A, const float* B, float* C, int N);

// Fills M with uniform random values in [-1, 1]. A fixed seed makes every run
// reproducible, which helps when debugging a mismatch.
void fill_random(std::vector<float>& M, unsigned seed);

struct CompareResult {
    bool  passed;
    int   mismatches;       // number of elements outside the tolerance
    int   first_bad_index;  // -1 if every element matched
    float max_abs_error;
};

// Element-wise check: |test - ref| <= atol + rtol * |ref|.
// Floating-point addition is not associative, so a GPU summing in a different
// order than the CPU will not produce bit-identical results; we need a tolerance.
CompareResult compare_matrices(const float* ref, const float* test, int N,
                               float atol = 1e-3f, float rtol = 1e-3f);

struct TimingStats {
    double min_ms;
    double median_ms;
    double mean_ms;
    int samples;
};

// Produces stable summary statistics from repeated timing samples.
// The vector is passed by value because calculating the median requires sorting.
TimingStats summarize_timings(std::vector<double> timings_ms);

// ---------------------------------------------------------------------------
// GPU implementations (defined in .cu files).
//
// These are ordinary host functions, so callers (main.cpp, tests) stay plain
// C++ and never need nvcc. Each one takes host pointers, does all device
// memory management and copies internally, writes the result into C, and
// returns statistics for the kernel alone in milliseconds.
// ---------------------------------------------------------------------------

// Returns false when three N x N device buffers will not fit comfortably in
// currently available GPU memory.
bool cuda_has_enough_memory(int N);

// One thread computes one element of C, reading A and B straight from global memory.
TimingStats matmul_gpu_naive(const float* A, const float* B, float* C, int N,
                            int warmup_runs = 1, int measured_runs = 1);

// One thread still computes one output element, but threads in a block
// cooperatively cache tiles of A and B in shared memory for reuse.
TimingStats matmul_gpu_tiled(const float* A, const float* B, float* C, int N,
                            int warmup_runs = 1, int measured_runs = 1);
