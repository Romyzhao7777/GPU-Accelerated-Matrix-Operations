#include "matrix.hpp"
#include "cuda_check.cuh"

#include <cstddef>

namespace {

constexpr int TILE_SIZE = 16;

// Each block computes one TILE_SIZE x TILE_SIZE tile of C.
//
// The naive kernel reads every A and B operand directly from global memory.
// Here, the threads in a block cooperate to load a tile of A and a tile of B
// into shared memory. Shared memory is on-chip and much faster than global
// memory, so every loaded value can be reused by 16 threads in the block.
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    const int tile_count = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * TILE_SIZE + tx;
        const int b_row = tile * TILE_SIZE + ty;

        // Every thread loads one value from A and one from B. Partial tiles at
        // the matrix edges are padded with zero, so the multiply loop below
        // needs no special-case bounds checks.
        tile_A[ty][tx] = (row < N && a_col < N)
                             ? A[row * N + a_col]
                             : 0.0f;
        tile_B[ty][tx] = (b_row < N && col < N)
                             ? B[b_row * N + col]
                             : 0.0f;

        // No thread may consume a tile until every thread has finished loading it.
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        // No thread may overwrite shared memory with the next tile while
        // another thread is still using the current tile.
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

}  // namespace

TimingStats matmul_gpu_tiled(const float* A, const float* B, float* C, int N,
                            int warmup_runs, int measured_runs) {
    if (warmup_runs < 0) {
        warmup_runs = 0;
    }
    if (measured_runs < 1) {
        measured_runs = 1;
    }

    const std::size_t bytes = static_cast<std::size_t>(N) * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                    (N + TILE_SIZE - 1) / TILE_SIZE);

    // Keep one-time kernel setup and the GPU's idle-to-active transition
    // outside the measured samples.
    for (int i = 0; i < warmup_runs; ++i) {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> timings_ms;
    timings_ms.reserve(measured_runs);
    for (int i = 0; i < measured_runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        timings_ms.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return summarize_timings(timings_ms);
}
