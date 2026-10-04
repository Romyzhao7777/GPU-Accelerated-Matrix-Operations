cmake_minimum_required(VERSION 3.18)

# Stage 1 is CPU-only. CUDA will be added to LANGUAGES in Stage 2.
project(GPUAcceleratedMatrixOps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
)
target_include_directories(matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(matmul_bench src/main.cpp)
target_link_libraries(matmul_bench PRIVATE matmul)

enable_testing()
add_executable(matmul_tests tests/test_correctness.cpp)
target_link_libraries(matmul_tests PRIVATE matmul)
add_test(NAME correctness COMMAND matmul_tests)
