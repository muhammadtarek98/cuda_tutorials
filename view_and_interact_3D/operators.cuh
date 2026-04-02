#pragma once
#include <vector_types.h>

__host__ __device__ void operator+=(float3 &a, float3 &b);
__host__ __device__ void operator+=(float3 &a, float b);
__host__ __device__ void operator+=(int3 &a, int3 &b);
__host__ __device__ void operator+=(int3 &a, int b);
__host__ __device__ void operator-=(float3 &a, float3 &b);
__host__ __device__ void operator-=(float3 &a, float b);
__host__ __device__ void operator-=(int3 &a, int3 &b);
__host__ __device__ void operator-=(int3 &a, int b);
__host__ __device__ float3 operator+(float3 a, float b);
__host__ __device__ float3 operator+(float b, float3 a);
__host__ __device__ float3 operator+(float3 a, float3 b);
__host__ __device__ int3 operator+(int3 a, int b);
__host__ __device__ int3 operator+(int b, int3 a);
__host__ __device__ int3 operator+(int3 a, int3 b);
__host__ __device__ float3 operator-(float3 a, float b);
__host__ __device__ float3 operator-(float b, float3 a);
__host__ __device__ float3 operator-(float3 a, float3 b);
__host__ __device__ int3 operator-(int3 a, int b);
__host__ __device__ int3 operator-(int b, int3 a);
__host__ __device__ int3 operator-(int3 a, int3 b);
__host__ __device__ float operator*(float a, float3 b);
__host__ __device__ float3 operator*(float3 a, float3 b);
__host__ __device__ float3 operator*(float3 a, float b);
__host__ __device__ float3 operator/(float3 a, float3 b);
__host__ __device__ float3 operator/(float3 a, float b);
__host__ __device__ uint3 operator*(uint3 a, float b);