cmake_minimum_required(VERSION 4.1)
project(laplace_equation_for_temperature_equalibrim LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CUDA)
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

set(SOURCES
        main.cpp
        src/graphics_utils.cpp
        src/intersection.cpp
        src/kernel.cu
)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${GLEW_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/headers
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE     ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        CUDA::cudart)
set_source_files_properties(src/graphics_utils.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)


