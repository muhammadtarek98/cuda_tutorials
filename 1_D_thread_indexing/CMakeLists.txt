cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "75")
project(1_D_thread_indexing CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
