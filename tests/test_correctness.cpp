// Correctness tests. Exit code 0 means every test passed, so `ctest` can run it.
//
// Stage 1 validates the CPU baseline itself, because every GPU kernel will
// later be checked against it. A wrong reference would make every later
// comparison meaningless.

#include "matrix.hpp"

#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void report(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) {
        ++g_failures;
    }
}

// Deliberately simple i-j-k triple loop with a double accumulator. It is slow,
// but it is the formula straight from the definition, so it serves as an
// independent check on the optimized loop order in matmul_cpu.
void matmul_textbook(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<double>(A[i * N + k]) * B[k * N + j];
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

void test_known_2x2() {
    // [1 2]   [5 6]   [19 22]
    // [3 4] x [7 8] = [43 50]
    const std::vector<float> A = {1, 2, 3, 4};
    const std::vector<float> B = {5, 6, 7, 8};
    const std::vector<float> expected = {19, 22, 43, 50};
    std::vector<float> C(4);

    matmul_cpu(A.data(), B.data(), C.data(), 2);
    report("2x2 hand-computed result", C == expected);
}

void test_identity() {
    const int N = 64;
    std::vector<float> A(N * N), I(N * N, 0.0f), C(N * N);
    fill_random(A, 1);
    for (int i = 0; i < N; ++i) {
        I[i * N + i] = 1.0f;
    }

    // Multiplying by 1 and adding 0 are exact in floating point,
    // so A * I must reproduce A bit-for-bit.
    matmul_cpu(A.data(), I.data(), C.data(), N);
    report("A * I == A (exact)", C == A);
}

void test_against_textbook(int N) {
    std::vector<float> A(N * N), B(N * N), C(N * N), ref(N * N);
    fill_random(A, 42);
    fill_random(B, 43);

    matmul_textbook(A.data(), B.data(), ref.data(), N);
    matmul_cpu(A.data(), B.data(), C.data(), N);

    const CompareResult r = compare_matrices(ref.data(), C.data(), N);
    char name[96];
    std::snprintf(name, sizeof(name), "matmul_cpu vs textbook reference, N=%d (max abs err %.2e)",
                  N, r.max_abs_error);
    report(name, r.passed);
}

void test_compare_detects_error() {
    // A checker that always says "pass" is worse than no checker,
    // so confirm it actually flags a wrong element.
    const int N = 32;
    std::vector<float> A(N * N);
    fill_random(A, 7);
    std::vector<float> B = A;
    B[5 * N + 9] += 0.5f;

    const CompareResult r = compare_matrices(A.data(), B.data(), N);
    report("compare_matrices flags an injected error",
           !r.passed && r.mismatches == 1 && r.first_bad_index == 5 * N + 9);
}

void test_gpu_naive(int N) {
    std::vector<float> A(N * N), B(N * N), ref(N * N), C(N * N);
    fill_random(A, 100);
    fill_random(B, 101);

    matmul_cpu(A.data(), B.data(), ref.data(), N);
    matmul_gpu_naive(A.data(), B.data(), C.data(), N);

    const CompareResult r = compare_matrices(ref.data(), C.data(), N);
    char name[96];
    std::snprintf(name, sizeof(name), "naive GPU vs CPU, N=%d (max abs err %.2e)",
                  N, r.max_abs_error);
    report(name, r.passed);
}

}  // namespace

int main() {
    std::printf("Running correctness tests...\n");

    test_known_2x2();
    test_identity();
    // 1 and 200 are not multiples of the future CUDA block size (16), which
    // is useful once the same sizes are used to test kernel boundary checks.
    test_against_textbook(1);
    test_against_textbook(200);
    test_against_textbook(512);
    test_compare_detects_error();

    // Sizes chosen around the 16x16 block size: smaller than one block (1, 15),
    // exactly one block (16), one block plus a sliver (17), and non-multiples
    // (200) where partial blocks at the right and bottom edges must be masked off.
    for (const int N : {1, 15, 16, 17, 200, 512, 1024}) {
        test_gpu_naive(N);
    }

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) FAILED.\n", g_failures);
    return 1;
}
