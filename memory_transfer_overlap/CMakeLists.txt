cmake_minimum_required(VERSION 4.1)
project(memory_transfer_overlap LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
add_executable(memory_transfer_overlap main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
