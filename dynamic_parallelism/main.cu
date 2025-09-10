#include<bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void kernel(int size,int depth)
{
    printf("depth: %d - thread index:%d \n",depth,threadIdx.x);
    if (size==1){return;}
    if (threadIdx.x==0)
    {
        kernel<<<1,size/2>>>(size/2,depth+1);
    }
}
int main(int argc,const char *argv[])
{
    kernel<<<1,16>>>(16,0);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    /**/
    return 0;
}