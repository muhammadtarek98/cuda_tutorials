cmake_minimum_required(VERSION 4.1)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
project(warp_shuffling LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(warp_shuffling main.cu)

set_target_properties(warp_shuffling PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
