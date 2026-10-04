#include "matrix.hpp"
#include "cuda_check.cuh"

#include <cstddef>
#include <limits>

bool cuda_has_enough_memory(int N) {
    const std::size_t n = static_cast<std::size_t>(N);
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        return false;
    }

    const std::size_t elements = n * n;
    constexpr std::size_t bytes_per_element =
        3 * sizeof(float);  // d_A, d_B, and d_C
    if (elements > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
        return false;
    }
    const std::size_t required = elements * bytes_per_element;

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    // Keep 10% of currently free memory as headroom for the CUDA context and
    // other runtime allocations instead of attempting a borderline cudaMalloc.
    return required <= free_bytes - free_bytes / 10;
}
