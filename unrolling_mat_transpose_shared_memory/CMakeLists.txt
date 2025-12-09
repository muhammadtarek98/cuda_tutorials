cmake_minimum_required(VERSION 4.1)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
project(unrolling_mat_transpose_shared_memory CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(unrolling_mat_transpose_shared_memory main.cu)

set_target_properties(unrolling_mat_transpose_shared_memory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
