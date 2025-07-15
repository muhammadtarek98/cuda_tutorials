#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
__global__ void kernel()
{
    printf( "Hello, World!\n");

}
int main()
{
    int nx=16,ny=4;
    dim3 block(nx/2,ny/2);
    dim3 grid(nx/block.x,ny/block.y);
    kernel<<<grid,block  >>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}