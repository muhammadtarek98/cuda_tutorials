cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "75")
project(hello_world CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(hello_world main.cu)

set_target_properties(hello_world PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
