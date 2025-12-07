cmake_minimum_required(VERSION 4.0)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)

project(shared_mem_row_major_access LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 20)

add_executable(shared_mem_row_major_access main.cu)

set_target_properties(shared_mem_row_major_access PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
