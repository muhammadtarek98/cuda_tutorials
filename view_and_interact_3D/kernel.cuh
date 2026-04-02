#pragma once
#define TX_2D 32
#define TY_2D 32
#define TY 8
#define TX 8
#define TZ 8
#define NUM 20
#include "operations.cuh"
#include "operators.cuh"
#include "ray_structure.cuh"
struct uchar4;
struct int3;
struct float4;
__device__ float compute_and_store(int c,int r,int s,int id,int3 vol_size,float4 params);
__global__ void kernel_render(uchar4 *d_out,float *d_vol,int w,int h,int3 vol_size,int method,float zs,float theta,float threshold,float dist);
__global__ void volume_kernel(float *d_vol,int3 vol_size,int id,float4 params);
void kernel_launch(uchar4 *d_out,float *d_vol,int w,int h,int3 vol_size,int method,int zs,float theta,float threshold,float dist);
void volume_kernel_launch(float *d_vol,int3 vol_size,int id,float4 params);
