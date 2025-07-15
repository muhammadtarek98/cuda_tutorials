#include <iostream>
__global__ void print_thread_idx()
{
    printf("threadIdx.x:%d threadIdx.y:%d threadIdx.z:%d\n",threadIdx.x,threadIdx.y,threadIdx.z);
}
int main()
{
    int nx=16,ny=16;
    dim3 block(8,8),grid(nx/block.x,ny/block.y);
    print_thread_idx<<<grid,block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}