cmake_minimum_required(VERSION 4.1)
project(async_functions LANGUAGES CXX C CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(async_functions main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
