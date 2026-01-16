cmake_minimum_required(VERSION 4.1)
project(streams_interdependcies LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
enable_language(CUDA)
add_executable(streams_interdependcies main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
