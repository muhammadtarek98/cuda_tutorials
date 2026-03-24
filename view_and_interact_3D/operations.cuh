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
