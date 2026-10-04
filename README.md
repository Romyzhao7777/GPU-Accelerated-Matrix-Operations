# GPU-Accelerated Matrix Operations

A small CUDA/C++ project that implements square matrix multiplication from
scratch and studies how GPU parallelism and memory access affect performance.

The project is being developed incrementally:

- [x] Single-threaded CPU baseline
- [x] Naive CUDA kernel
- [x] Shared-memory tiled CUDA kernel
- [ ] Repeated-run benchmark harness
- [ ] Final performance analysis and graph

Current status: **Stage 3 complete and verified on an NVIDIA Tesla T4**.

## Motivation

The goal is to understand the complete path from a CPU implementation to an
optimized CUDA implementation, rather than treating GPU code as a black box.
Each stage is checked against the CPU result before the next optimization is
introduced.

The project currently demonstrates:

- Row-major matrix storage in contiguous memory
- CUDA grid, block, and thread indexing
- Cooperative shared-memory tiling and synchronization
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
global memory.

### Shared-memory tiled CUDA kernel

The tiled kernel also uses `16 x 16` thread blocks, but each block computes one
`16 x 16` tile of the output matrix. Computation advances through the input
matrices one tile at a time:

1. Each thread loads one value from `A` and one value from `B` into shared
   memory.
2. `__syncthreads()` ensures that the complete tiles are available.
3. Each thread reuses the shared values to perform 16 multiply-add operations.
4. A second `__syncthreads()` prevents the next tile from overwriting data that
   another thread is still using.

Each block allocates two shared arrays:

```text
tile_A[16][16] + tile_B[16][16] = 2,048 bytes
```

Values outside a partial edge tile are loaded as zero. This allows every
thread to reach both synchronization barriers safely while keeping the inner
multiply loop free of boundary checks.

In the naive kernel, an input value may be fetched repeatedly from global
memory by many threads. Tiling loads it once per block and reuses it from
low-latency, on-chip shared memory. With a tile width of 16, the ideal operand
reuse factor is up to 16 within a block.

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
`2.10e-05`.

## Preliminary results

Environment:

- GPU: NVIDIA Tesla T4, 15 GB
- NVIDIA driver: 580.82.07
- CUDA compiler/runtime: CUDA 13.0
- Host compiler: GCC 13.3.0
- Build type: Release
- Platform: Google Colab
- Matrix data type: 32-bit floating point

| Matrix size | CPU (ms) | Naive (ms) | Tiled (ms) | CPU / Tiled | Naive / Tiled |
|---:|---:|---:|---:|---:|---:|
| 256 x 256 | 4.01 | 0.092 | 0.062 | 64.7x | 1.48x |
| 512 x 512 | 29.28 | 0.629 | 0.409 | 71.6x | 1.54x |
| 1024 x 1024 | 397.13 | 4.797 | 3.004 | 132.2x | 1.60x |
| 2048 x 2048 | 2484.47 | 74.932 | 31.719 | 78.3x | 2.36x |
| 4096 x 4096 | 25738.64 | 318.203 | 205.451 | 125.3x | 1.55x |

The tiled kernel was faster than the naive kernel at every tested size, with
an observed improvement of `1.48x` to `2.36x`.

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
│   ├── cuda_tiled.cu        # Shared-memory tiled CUDA kernel
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
./build/matmul_bench 256 512 1024 2048 4096
```

## Next stage

Stage 4 will replace the preliminary single-run timing with a proper benchmark
harness. It will run each implementation multiple times, separate warm-up from
measurement, report stable summary statistics, and handle sizes that are too
slow or exceed available memory.
