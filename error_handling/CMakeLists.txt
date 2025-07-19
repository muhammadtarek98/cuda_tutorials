cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(error_handling LANGUAGES C CXX CUDA)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)

add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

