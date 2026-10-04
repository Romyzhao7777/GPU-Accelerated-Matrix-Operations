#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Almost every CUDA runtime call returns a cudaError_t, and errors are easy to
// miss because nothing throws: a failed cudaMalloc just leaves a bad pointer,
// and a failed kernel launch silently produces garbage. Wrapping each call in
// CUDA_CHECK makes any failure stop the program with the file and line number.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in: %s\n",         \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call);  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
