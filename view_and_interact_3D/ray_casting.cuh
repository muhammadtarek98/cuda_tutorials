#pragma once
#include "ray_structure.cuh"
#include "operators.cuh"
#include "operations.cuh"
#include "slicing.cuh"
__device__ uchar4 ray_caster(float *d_vol,int3 vol_size,Ray ray,float dist);
