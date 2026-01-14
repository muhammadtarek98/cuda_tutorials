cmake_minimum_required(VERSION 4.1)
project(non_nuil_stream_with_async_functions LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
enable_language(CUDA)
add_executable(non_nuil_stream_with_async_functions main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
