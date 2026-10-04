cmake_minimum_required(VERSION 3.18)

# Which GPU generation(s) nvcc generates machine code for. "native" (CMake >= 3.24)
# detects the GPU in the build machine. Otherwise fall back to 75 (Turing, e.g.
# the Tesla T4 on Google Colab). Override with -DCMAKE_CUDA_ARCHITECTURES=86, etc.
# Must be set before project(), which is where CMake picks its own default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
endif()

project(GPUAcceleratedMatrixOps LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Benchmarks are meaningless in a Debug build, so default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

# Core library shared by the benchmark executable and the tests.
add_library(matmul STATIC
    src/cpu.cpp
    src/matrix_utils.cpp
    src/cuda_naive.cu
)
target_include_directories(matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(matmul_bench src/main.cpp)
target_link_libraries(matmul_bench PRIVATE matmul)

enable_testing()
add_executable(matmul_tests tests/test_correctness.cpp)
target_link_libraries(matmul_tests PRIVATE matmul)
add_test(NAME correctness COMMAND matmul_tests)
