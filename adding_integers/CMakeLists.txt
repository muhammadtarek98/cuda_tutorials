cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
project(adding_integers CUDA)
set(CMAKE_CUDA_STANDARD 20)

add_executable(adding_integers main.cu)

set_target_properties(adding_integers PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
