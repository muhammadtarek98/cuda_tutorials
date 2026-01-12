cmake_minimum_required(VERSION 4.1)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)
project(laplace_heat_2d  LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
