#pragma once
#include <cuda_runtime.h>
#include "operators.cuh"
#include "operations.cuh"
#include "ray_structure.cuh"
__device__ float planeSDF(float3 pos,float3 norm,float d);
__device__ bool ray_plane_intersect(Ray ray,float3 n,float dist,float *t);
__device__ uchar4 slice_shader(float *d_vol,int3 vol_size,Ray ray,float gain,float dist,float3 norm);