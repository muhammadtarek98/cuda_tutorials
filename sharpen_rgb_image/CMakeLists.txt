cmake_minimum_required(VERSION 3.18)
project(sharpen_rgb_image LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xcompiler -fPIC")
set(CUDA_SEPARABLE_COMPILATION ON)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
add_library(image_sharpen_kernels STATIC ./src/kernel.cu)
include_directories(headers)
add_executable(${PROJECT_NAME} main.cpp )
target_link_libraries(image_sharpen_kernels PRIVATE CUDA::cudart)
target_include_directories(image_sharpen_kernels PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} headers)
target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES} Threads::Threads image_sharpen_kernels CUDA::cudart)
target_include_directories(${PROJECT_NAME} PRIVATE headers/ ${X11_INCLUDE_DIR}  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)