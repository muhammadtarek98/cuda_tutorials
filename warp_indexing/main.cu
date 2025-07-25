#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
extern "C"{
__global__ void kernel()
{
    int thread_global_id=blockIdx.y*gridDim.x*blockDim.y+blockIdx.x*blockDim.x+threadIdx.x;
    int warp_id=threadIdx.x/32;
    int block_global_id=blockIdx.y*gridDim.x+blockIdx.x;
    printf("thread id:%d \tblock id.x:%d \tblock id.y:%d \tthread global id:%d \twarp id:%d \tblock global id:%d\n",
        threadIdx.x,blockIdx.x,blockIdx.y,thread_global_id,warp_id,block_global_id);
}
}
int main()
{
    dim3 block_size(42);
    dim3 grid_size(2,2);
    kernel<<<grid_size,block_size>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}