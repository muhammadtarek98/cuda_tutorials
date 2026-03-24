cmake_minimum_required(VERSION 4.1)
project(view_and_interact_3D LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
add_executable(view_and_interact_3D main.cu
        ray_structure.cu
        ray_structure.cuh
        slicing.cu
        slicing.cuh
        operators.cu
        operators.cuh
        operations.cu
        operations.cuh
        volume_render.cu
        volume_render.cuh
        ray_casting.cu
        ray_casting.cuh
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
set_target_properties(view_and_interact_3D PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
