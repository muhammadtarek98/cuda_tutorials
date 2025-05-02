cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER  "/usr/local/cuda-11.8/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "75")
project(matrix_multiplication CUDA)
option(ENABLE_MEMCHECK "Enable memory checking tools" ON)

set(CMAKE_CUDA_STANDARD 20)

add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared)
if (ENABLE_MEMCHECK)
    add_custom_target(
            run_memcheck
            COMMAND cuda-memcheck $<TARGET_FILE:${PROJECT_NAME}>
            COMMENT "Running cuda-memcheck..."
            VERBATIM
    )

    add_custom_target(
            run_valgrind
            COMMAND valgrind --leak-check=full $<TARGET_FILE:${PROJECT_NAME}>
            COMMENT "Running valgrind (host memory check)..."
            VERBATIM
    )
endif()