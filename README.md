# GPU-Accelerated Matrix Operations

Matrix multiplication implemented three ways — a CPU baseline, a naive CUDA
kernel, and a shared-memory tiled CUDA kernel — and benchmarked across matrix
sizes.

> Work in progress. Current status: **Stage 1 (CPU baseline + correctness tests)**.

## Build

Requires CMake >= 3.18 and a C++17 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
./build/matmul_tests            # correctness tests
./build/matmul_bench            # CPU timing for N = 256, 512, 1024
./build/matmul_bench 128 2048   # custom sizes
```
