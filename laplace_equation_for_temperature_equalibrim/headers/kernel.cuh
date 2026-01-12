#pragma once
#include "common_utils.h"
#include <cuda_runtime.h>
#define TX 32
#define TY 32
#define RAD 1
 int div_up(int a,int b);
__device__ unsigned char clip(int n);
__device__ float dist_sqr(int x,int y,int row,int col);
__device__ int idxclip(int idx,int idx_max);
__device__ int flatten(int col,int row,int width,int height);
__global__ void reset_kernel(float *d_temp,int w,int h,BC bc);
__global__ void temp_kernel(uchar4 *d_out,float *d_temp, int w,int h,BC bc);
 void kernel_launcher(uchar4 *d_out,float *d_temp, int w,int h,BC bc);
 void reset_temp(float *d_temp,int w,int h,BC bc);
