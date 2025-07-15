#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void print_details()
{
    printf("threadIdx.x: %d threadIdx.y: %d threadIdx.z: %d blockIdx.x: %d blockIdx.y: %d blockIdx.z: %d blockDim.x: %d blockDim.y: %d blockDim.z: %d gridDim.x: %d gridDim.y: %d gridDim.z: %d\n"
        ,threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);
}
int main()
{
    int nx=16,ny=16;
    dim3 block(nx/2,ny/2);
    dim3 grid(nx/block.x,ny/block.y);
    print_details<<<grid,block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}