cmake_minimum_required(VERSION 4.1)
project(view_and_interact_3D LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
add_executable(view_and_interact_3D main.cpp
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
        visualization_utils.cpp
        visualization_utils.h
        interactions.cpp
        interactions.h
        kernels_launch.h
        kernel.cu
        kernel.cuh
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart
        GL
        GLU
        glut GLEW::GLEW )
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)