# GPU-Accelerated Matrix Operations

A small CUDA/C++ project that implements square matrix multiplication from
scratch and studies how GPU parallelism and memory access affect performance.

The project is being developed incrementally:

- [x] Single-threaded CPU baseline
- [x] Naive CUDA kernel
- [x] Shared-memory tiled CUDA kernel
- [x] Repeated-run benchmark harness
- [x] Benchmark collection on an NVIDIA Tesla T4
- [x] Final performance analysis and graph

Current status: **Complete — all implementations are verified, benchmarked,
and documented**.

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
- Repeated CPU-versus-GPU performance measurement

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

## Benchmark methodology

- GPU kernels run twice for warm-up, followed by 10 measured runs.
- CPU runs five times through `N=1024`, three times at `N=2048`, and once at
  `N=4096` to keep the full benchmark practical.
- The table reports the median of the measured samples. The `N=4096` CPU value
  is a single sample rather than a true median.
- CPU time is measured with `std::chrono`.
- GPU time is measured with CUDA events around the kernel only.
- CUDA context initialization, device allocation, and host-device transfers
  are excluded from GPU kernel time.
- The program checks available GPU memory before allocating three device
  matrices and skips a size when insufficient memory is available.

The reported speedups are therefore **kernel speedups, not end-to-end
application speedups**.

## Benchmark results

Environment:

- GPU: NVIDIA Tesla T4, 15 GB
- CPU: Intel Xeon at 2.00 GHz, 1 core / 2 threads
- NVIDIA driver: 580.82.07
- CUDA compiler: nvcc 13.0.88
- Host compiler: GCC 13.3.0
- Build type: Release
- Platform: Google Colab
- Matrix data type: 32-bit floating point
- Benchmark date: October 4, 2026

| Matrix size | CPU (ms) | Naive (ms) | Tiled (ms) | CPU / Tiled | Naive / Tiled |
|---:|---:|---:|---:|---:|---:|
| 256 x 256 | 4.50 | 0.096 | 0.067 | 67.4x | 1.44x |
| 512 x 512 | 37.58 | 0.685 | 0.442 | 85.1x | 1.55x |
| 1024 x 1024 | 301.27 | 9.193 | 5.802 | 51.9x | 1.58x |
| 2048 x 2048 | 2362.35 | 40.365 | 26.174 | 90.3x | 1.54x |
| 4096 x 4096 | 27028.84 | 330.236 | 214.902 | 125.8x | 1.54x |

![Execution-time comparison](results/performance.svg)

## Performance analysis

### CPU versus GPU

Both CUDA kernels substantially outperform the single-threaded CPU baseline.
The tiled kernel ranges from `51.9x` to `125.8x` faster in kernel execution
time. The advantage generally becomes larger for bigger matrices because more
parallel work is available to fill the GPU.

This comparison is intentionally against the project's own single-threaded
CPU implementation. It is not a comparison with a multithreaded BLAS library
such as Intel MKL.

### Naive versus tiled CUDA

The tiled kernel was faster at every tested size. Its improvement over the
naive kernel remained between `1.44x` and `1.58x`, showing that shared-memory
reuse consistently reduced kernel time across this range.

The theoretical reduction in repeated global-memory reads does not translate
directly into an equal runtime improvement. The tiled kernel also pays for
shared-memory loads, index calculations, and two synchronization barriers per
tile. In addition, the naive kernel already benefits from coalesced reads of
`B`, broadcast reads of `A`, and the GPU's hardware caches. The observed
roughly `1.5x` improvement is therefore plausible and, importantly,
repeatable across the tested sizes.

### Scaling behavior

Matrix multiplication performs roughly `2N^3` floating-point operations, so
doubling `N` ideally increases the work by about eight times. The overall
timing trend follows this cubic growth, although individual ratios vary due to
cache behavior, GPU clock state, and the shared Google Colab environment.
This is why the benchmark reports medians instead of relying on a single GPU
measurement.

## Measurement limitations

- GPU timings cover kernel execution only; allocation and PCIe transfers are
  excluded.
- The CPU implementation is single-threaded and is not a vendor-optimized BLAS
  implementation.
- Google Colab hardware load and clock behavior can vary between sessions.
- The `4096 x 4096` CPU result uses one run because each multiplication takes
  about 27 seconds.
- Results apply to the listed Tesla T4 environment and should not be assumed
  for other GPU architectures.

The machine-readable results, including run counts, are stored in
[`results/tesla_t4.csv`](results/tesla_t4.csv).

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
│   ├── cuda_utils.cu        # GPU memory availability check
│   ├── main.cpp             # Correctness check and timing output
│   └── matrix_utils.cpp     # Random initialization and result comparison
├── tests/
│   └── test_correctness.cpp # CPU and CUDA correctness tests
├── results/
│   ├── performance.svg      # Execution-time comparison graph
│   └── tesla_t4.csv         # Reproducible benchmark data
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

Run the default matrix sizes (`256`, `512`, `1024`, and `2048`):

```bash
./build/matmul_bench
```

Pass custom square matrix sizes as command-line arguments:

```bash
./build/matmul_bench 256 512 1024 2048 4096
```

## Possible extensions

The current scope is intentionally small. Potential follow-up experiments
include:

- Comparing different CUDA block and tile sizes
- Matrix addition or transpose
- Comparing against cuBLAS
- Profiling with Nsight Compute
- Studying memory coalescing
- Pinned host memory
- CUDA streams and overlapping transfer with computation
