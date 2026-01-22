cmake_minimum_required(VERSION 4.1)
project(atomic_parrallel_dot_product LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
add_executable(atomic_parrallel_dot_product main.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
set_target_properties(atomic_parrallel_dot_product PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
