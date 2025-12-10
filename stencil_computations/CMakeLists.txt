cmake_minimum_required(VERSION 4.1)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
project(stencil_computations LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(stencil_computations main.cu)

set_target_properties(stencil_computations PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
