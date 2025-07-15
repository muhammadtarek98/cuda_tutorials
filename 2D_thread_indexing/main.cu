#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
__global__ void kernel(int *a)
{
    int tidx=blockDim.x*threadIdx.y+threadIdx.x;
    int threads_per_block=blockDim.x*blockDim.y;
    int block_offset=blockIdx.x*threads_per_block;
    int threads_per_row=threads_per_block*gridDim.x;
    int row_offset=threads_per_row*blockIdx.y;
    int tidy=row_offset+block_offset+tidx;
    printf("blockidx.x: %d blockidx.y: %d threadidx.x:%d threadid.y:%d value:%d\n",blockIdx.x ,blockIdx.y,
        tidx ,tidy,a[tidy]);
}
int main()
{
    dim3 block (2,2),grid(2,2);
    int n=16,nbytes=sizeof(int)*n;
    int h_arr[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    int *d_arr;
    cudaMalloc((void**)&d_arr,nbytes);
    cudaMemcpy(d_arr,h_arr,nbytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block>>>(d_arr);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}