#pragma once
#include <cuda_runtime.h>
#include "operators.cuh"
#include "operations.cuh"
#include "ray_structure.cuh"
__device__ int clip(int n,int n_max,int n_min);
__device__ float density(float *d_vol,int3 vol_size,float3 pos);
__device__ int3 posToIdx(float3 pos,int3 vol_size);
__host__ __device__ float fracf(float v);
__host__ __device__ float3 fracf(float3 v);
__device__ int flatten(int3 index, int3 volSize);
__device__ float planeSDF(float3 pos,float3 norm,float d);
__device__ bool ray_plane_intersect(Ray ray,float3 n,float dist,float *t);
__device__ uchar4 slice_shader(float *d_vol,int3 vol_size,Ray ray,float gain,float dist,float3 norm);