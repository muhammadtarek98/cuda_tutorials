cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "75")
project(threads_access  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(threads_access main.cu)

set_target_properties(threads_access PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
