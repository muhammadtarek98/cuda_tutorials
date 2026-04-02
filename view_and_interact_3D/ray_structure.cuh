#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include "operators.cuh"
typedef struct
{
    float3 orign;
    float3 dir;
}Ray;
__device__ float3 param_ray(Ray r,float t);
__device__ bool intersection(Ray r,float3 box_min,float3 box_max,float *t_near,float *t_far);