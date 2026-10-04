# GPU-Accelerated Matrix Operations

A small CUDA/C++ project that implements square matrix multiplication from
scratch and studies how GPU parallelism and memory access affect performance.

The project is being developed incrementally:

- [x] Single-threaded CPU baseline
- [x] Naive CUDA kernel
- [ ] Shared-memory tiled CUDA kernel
- [ ] Repeated-run benchmark harness
- [ ] Final performance analysis and graph

Current status: **Stage 2 complete and verified on an NVIDIA Tesla T4**.

## Motivation

The goal is to understand the complete path from a CPU implementation to an
optimized CUDA implementation, rather than treating GPU code as a black box.
Each stage is checked against the CPU result before the next optimization is
introduced.

The project currently demonstrates:

- Row-major matrix storage in contiguous memory
- CUDA grid, block, and thread indexing
- Host and device memory allocation and transfers
- Kernel timing with CUDA events
- Floating-point correctness checks
- Basic CPU-versus-GPU performance comparison

## Implementations

### CPU baseline

The CPU computes:

```text
C[i, j] = sum(A[i, k] * B[k, j]), for k = 0 ... N - 1
```

It uses an `i-k-j` loop order. The innermost loop walks through contiguous rows
of `B` and `C`, which has better cache behavior than the textbook `i-j-k`
ordering. This provides a more reasonable single-threaded CPU baseline.

### Naive CUDA kernel

The output matrix is divided into a two-dimensional grid of `16 x 16` thread
blocks. Each CUDA thread computes one output element:

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

The grid dimensions are rounded up so matrices do not need to be multiples of
16. Threads outside the matrix return without reading or writing memory.

Each active thread performs the full dot product for its output element. This
exposes substantial parallelism, but repeatedly reads the same values from
global memory. The next stage will reduce that redundant traffic with
shared-memory tiling.

## Correctness

The CPU result is the reference. CUDA output is checked element by element
using combined absolute and relative tolerances:

```text
|GPU - CPU| <= absolute_tolerance + relative_tolerance * |CPU|
```

Tests cover hand-computed multiplication, multiplication by the identity
matrix, an independent textbook CPU implementation, and CUDA sizes around the
`16 x 16` block boundary: `1`, `15`, `16`, `17`, `200`, `512`, and `1024`.

All tests pass on the Tesla T4. The largest observed absolute difference was
`1.91e-05`.

## Preliminary results

Environment:

- GPU: NVIDIA Tesla T4, 15 GB
- NVIDIA driver: 580.82.07
- CUDA compiler/runtime: CUDA 13.0
- Host compiler: GCC 13.3.0
- Build type: Release
- Platform: Google Colab
- Matrix data type: 32-bit floating point

| Matrix size | CPU time (ms) | Naive CUDA kernel (ms) | Kernel speedup |
|---:|---:|---:|---:|
| 256 x 256 | 5.86 | 0.091 | 64.7x |
| 512 x 512 | 35.68 | 0.631 | 56.6x |
| 1024 x 1024 | 281.39 | 4.991 | 56.4x |
| 2048 x 2048 | 2255.21 | 63.262 | 35.6x |

These are preliminary single-run measurements. CPU time is measured with
`std::chrono`, while GPU time is measured with CUDA events around the kernel
only. GPU allocation and host-device transfers are deliberately excluded, so
the reported speedup is **kernel speedup, not end-to-end application speedup**.
Stage 4 will add repeated runs and more robust statistics.

## Project structure

```text
.
├── include/
│   ├── cuda_check.cuh       # CUDA runtime error-checking macro
│   └── matrix.hpp           # Shared CPU/GPU interfaces
├── src/
│   ├── cpu.cpp              # CPU matrix multiplication
│   ├── cuda_naive.cu        # One-thread-per-output CUDA kernel
│   ├── main.cpp             # Correctness check and timing output
│   └── matrix_utils.cpp     # Random initialization and result comparison
├── tests/
│   └── test_correctness.cpp # CPU and CUDA correctness tests
├── CMakeLists.txt
└── README.md
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- CMake 3.18 or newer
- C++17 compiler supported by the installed CUDA Toolkit

## Build

```bash
git clone https://github.com/Romyzhao7777/GPU-Accelerated-Matrix-Operations.git
cd GPU-Accelerated-Matrix-Operations
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake detects the native CUDA architecture when supported. It can also be set
explicitly; for example, compute capability 7.5 for a Tesla T4:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75
```

## Run

Run all correctness tests:

```bash
./build/matmul_tests
```

Run the default matrix sizes (`256`, `512`, and `1024`):

```bash
./build/matmul_bench
```

Pass custom square matrix sizes as command-line arguments:

```bash
./build/matmul_bench 256 512 1024 2048
```

## Next stage

Stage 3 will implement a shared-memory tiled CUDA kernel. Threads in a block
will cooperatively load tiles of `A` and `B`, synchronize with
`__syncthreads()`, reuse each loaded value for multiple multiply-add
operations, and compare the result and execution time with the naive kernel.
