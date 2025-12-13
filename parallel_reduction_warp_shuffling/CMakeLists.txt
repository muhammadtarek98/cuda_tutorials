cmake_minimum_required(VERSION 4.1)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
project(parallel_reduction_warp_shuffling LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(parallel_reduction_warp_shuffling main.cu)

set_target_properties(parallel_reduction_warp_shuffling PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
