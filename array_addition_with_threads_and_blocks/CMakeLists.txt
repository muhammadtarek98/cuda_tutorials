cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
project(array_addition_with_threads_and_blocks CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(array_addition_with_threads_and_blocks main.cu)

set_target_properties(array_addition_with_threads_and_blocks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
