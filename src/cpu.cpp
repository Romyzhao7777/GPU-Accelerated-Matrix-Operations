#include "matrix.hpp"

#include <cstddef>

// Single-threaded CPU matrix multiplication: C[i][j] = sum_k A[i][k] * B[k][j].
//
// Loop order is i-k-j instead of the textbook i-j-k. Both do the same 2*N^3
// floating-point operations, but in i-j-k the inner loop walks down a column of
// B (stride N floats), which misses the cache on almost every access. In i-k-j
// the inner loop walks along a row of B and a row of C (stride 1), so memory is
// read sequentially and the compiler can auto-vectorize it. This keeps the CPU
// baseline honest: GPU speedups are measured against a reasonable CPU version,
// not an artificially slow one.
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    const std::size_t n = static_cast<std::size_t>(N);

    for (std::size_t idx = 0; idx < n * n; ++idx) {
        C[idx] = 0.0f;
    }

    for (std::size_t i = 0; i < n; ++i) {
        float* c_row = C + i * n;
        for (std::size_t k = 0; k < n; ++k) {
            const float a_ik = A[i * n + k];
            const float* b_row = B + k * n;
            for (std::size_t j = 0; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
