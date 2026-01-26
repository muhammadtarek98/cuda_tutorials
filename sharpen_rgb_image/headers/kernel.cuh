#pragma once
#define TX 32
#define TY 32
#define RAD 1
#include <cuda_runtime.h>
struct uchar4;
__device__ unsigned char clip(int n);
__device__ int clip_idx(int idx,int idx_mx);
__device__ int flatten(int col,int row,int w,int h);
__global__ void normal_sharpen_kernel(uchar4 *out, const uchar4 *in,const float *filter,int w,int h);
__global__ void single_shared_mem_sharpen_kernel(uchar4 *out, const uchar4 *in,const float *filter,int w,int h);
void sharpen_image(uchar4 *img_ptr,int w,int h,bool use_shared_mem_imp);
int div_up(int a,int b);