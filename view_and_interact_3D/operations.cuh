#pragma once
#include "operators.cuh"
#include <cmath>
__host__ __device__ float dot(float3 x,float3 y);
__host__ __device__ float dot(int3 x,int3 y);
__host__ __device__ float dot(uint3 x,uint3 y);
__host__ __device__ float dot(uint3 x,uint3 y);
__host__ __device__ float3 normalize(float3 x);
__host__ __device__ float length(float3 vector);
__host__ __device__ float length(float4 vector);
__host__ __device__ float dot(float4 x,float4 y);
__host__ __device__ int div_up(int a,int b);
__host__ __device__ float3 srcIdxToPos(int c,int r,int w,int h,float zs);
__host__ __device__ int3 posToVolIdx(float3 pos,int3 vol_size);
__host__ __device__ int flatten(int3 idx,int3 vol_size);
__device__ int clip(int n,int n_max,int n_min);
__device__ float density(float *d_vol,int3 vol_size,float3 pos);
__host__ __device__ float fracf(float v);
__host__ __device__ float3 fracf(float3 v);
__host__ __device__ int flatten(int3 index, int3 volSize);
__host__ __device__ float3 yRotate(float3 pos, float theta);