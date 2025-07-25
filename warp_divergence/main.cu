#include <iostream>
__global__ void kernel_1()
{
    int gid=threadIdx.x+blockIdx.x*blockDim.x;
    int warp_id=gid/32;
    float a=0.0,b=0.0;
    if (warp_id%2==0)
    {
        a=100.0,b=50.0;
    }
    else
    {
        a=200.0,b=500.0;
    }
}
__global__ void kernel_2()
{
    int gid=threadIdx.x+blockIdx.x*blockDim.x;
    float a=0.0,b=0.0;
    if (gid%2==0)
    {
        a=100.0,b=50.0;
    }
    else
    {
        a=200.0,b=500.0;
    }
}
int main()
{
    int size=1<<22;
    dim3 block_size(128);
    dim3 grid_size((size+block_size.x-1)/block_size.x);
    kernel_1<<<grid_size,block_size>>>();
    cudaDeviceSynchronize();
    kernel_2<<<grid_size,block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}