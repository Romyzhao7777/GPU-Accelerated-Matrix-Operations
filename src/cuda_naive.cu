#include "matrix.hpp"
#include "cuda_check.cuh"

#include <cstddef>

namespace {

// 16 x 16 = 256 threads per block: a multiple of the 32-thread warp size and
// well under the 1024-threads-per-block hardware limit.
constexpr int BLOCK_SIZE = 16;

// __global__ marks a kernel: a function that runs on the GPU and is launched
// from the host. Every thread runs this same code; the only thing that differs
// between threads is their built-in indices (blockIdx, threadIdx).
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int N) {
    // Global 2D position of this thread in the whole grid:
    //   which block we are in * threads per block + our position inside the block.
    // x maps to columns and y to rows, so consecutive threads in a warp
    // (consecutive threadIdx.x) handle consecutive columns of C.
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // The grid is rounded up to whole blocks, so when N is not a multiple of
    // BLOCK_SIZE some threads fall outside the matrix and must do nothing.
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // Every operand comes from global memory (DRAM). Across the whole
            // kernel each element of A and B is re-read N times by different
            // threads -- this redundancy is what the tiled kernel will remove.
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

}  // namespace

float matmul_gpu_naive(const float* A, const float* B, float* C, int N) {
    const std::size_t bytes = static_cast<std::size_t>(N) * N * sizeof(float);

    // 1. Allocate device (GPU) memory. Host and device have separate address
    //    spaces, so the kernel cannot read the host arrays directly.
    //    The very first CUDA call also creates the CUDA context (driver setup),
    //    which can take 100+ ms; it happens here, outside the timed region.
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 2. Copy the inputs host -> device.
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    // 3. Launch configuration: 2D blocks of 16x16 threads, and enough blocks
    //    to cover the N x N output. (N + BLOCK_SIZE - 1) / BLOCK_SIZE is
    //    integer ceiling division, e.g. N = 200 -> 13 blocks (208 threads) per side.
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up launch, not timed. The first launch of a kernel pays one-time
    // costs (e.g. loading the kernel code onto the GPU), which would otherwise
    // inflate the measurement.
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());       // catches invalid launch configurations
    CUDA_CHECK(cudaDeviceSynchronize());  // catches errors raised while the kernel ran

    // 4. Timed launch. Kernel launches are asynchronous: the <<<>>> call returns
    //    to the CPU immediately, before the GPU has finished. A CPU timer around
    //    it would mostly measure launch overhead. CUDA events are timestamps
    //    recorded by the GPU itself in its command stream, so the time between
    //    them is the actual kernel execution time.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));  // block the CPU until the GPU reaches `stop`

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // 5. Copy the result device -> host. cudaMemcpy waits for prior GPU work to finish.
    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return kernel_ms;
}
