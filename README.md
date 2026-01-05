# CUDA Tutorials

A comprehensive collection of CUDA programming tutorials covering various parallel programming concepts, memory models, optimization techniques, and performance patterns for NVIDIA GPUs.

## Overview

This repository contains practical examples and tutorials demonstrating CUDA programming fundamentals and advanced techniques. Each tutorial is self-contained and focuses on specific aspects of GPU programming, making it easy to learn and understand CUDA concepts incrementally.

## Tutorials

### Getting Started
- **hello_world** - Basic CUDA kernel execution with grid and block configuration
- **adding_integers** - Simple integer addition using CUDA
- **device_properties** - Query and display GPU device properties

### Thread and Block Management
- **1_D_thread_indexing** - Understanding 1D thread indexing in CUDA
- **2D_thread_indexing** - Working with 2D thread blocks and grids
- **threads_access** - Thread access patterns and organization
- **block_grid_access** - Accessing data using block and grid dimensions
- **warp_indexing** - Understanding warp-level thread organization

### Array Operations
- **vector_addition_block_style** - Vector addition using block-level parallelism
- **array_addition_thread_style** - Array addition with thread-level parallelism
- **array_addition_with_threads_and_blocks** - Combining threads and blocks for array operations
- **array_summation_profiling** - Performance profiling for array summation

### Memory Management
- **cuda_memory_model** - Overview of CUDA memory hierarchy
- **shared_memory** - Using shared memory for performance optimization
- **shared_memory_access** - Optimizing shared memory access patterns
- **shared_mem_row_major_access** - Row-major access patterns in shared memory
- **shared_memory_padding** - Using padding to avoid bank conflicts
- **unified_memory** - Unified memory for simplified memory management
- **pinned_memory** - Using page-locked host memory for faster transfers
- **zero_copy_memory** - Direct GPU access to host memory
- **global_memory_write_operations** - Optimizing global memory writes
- **memory_access_patterns** - Understanding efficient memory access patterns

### Matrix Operations
- **matrix_multiplication** - Basic matrix multiplication on GPU
- **matrix_transpose** - Matrix transposition implementations
- **matrix_transpose_shared_memory** - Using shared memory for matrix transpose
- **matrix_transpose_padded_shared_memory** - Padded shared memory for optimized transpose
- **matrix_transpose_with_padded_shared_memory** - Enhanced matrix transpose with padding

### Optimization Techniques
- **unrolling** - Loop unrolling for improved performance
- **complete_unrolling** - Complete loop unrolling techniques
- **warp_unrolling** - Warp-level loop unrolling
- **unrolling_mat_transpose_shared_memory** - Unrolled matrix transpose with shared memory
- **register_usage** - Optimizing register usage in kernels
- **template_parameters** - Using C++ templates for flexible kernel code

### Warp-Level Programming
- **warp_divergence** - Understanding and managing warp divergence
- **warp_shuffling** - Using warp shuffle operations for communication

### Parallel Reduction
- **parallel_reduction_with_shared_memory** - Reduction using shared memory
- **parallel_reduction_warp_shuffling** - Reduction with warp shuffle instructions
- **divergence_in_parallel_reduction** - Managing divergence in reduction operations
- **parallel_reduction_using_dynamic_parallelism** - Reduction with dynamic parallelism

### Advanced Features
- **dynamic_parallelism** - Launching kernels from within kernels
- **synchronization** - Thread synchronization mechanisms
- **error_handling** - Proper CUDA error handling techniques

### Data Structures and Patterns
- **AOS_VS_SOA** - Array of Structures vs Structure of Arrays comparison
- **stencil_algorithm** - Stencil computation patterns
- **stencil_computations** - Advanced stencil operations

## Building and Running

Each tutorial directory contains:
- `main.cu` - CUDA source code
- `CMakeLists.txt` - CMake build configuration

To build and run a tutorial:

```bash
cd <tutorial_name>
mkdir build
cd build
cmake ..
make
./<executable_name>
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- CMake build system
- C++ compiler with C++11 support or higher

## Tutorial Structure

Each tutorial is designed to be:
- **Self-contained** - Can be built and run independently
- **Focused** - Demonstrates a specific concept or technique
- **Practical** - Includes working code examples
- **Educational** - Organized progressively from basic to advanced topics
