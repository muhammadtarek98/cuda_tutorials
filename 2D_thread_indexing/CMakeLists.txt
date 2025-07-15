cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
project(2D_thread_indexing CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(2D_thread_indexing main.cu)

set_target_properties(2D_thread_indexing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
