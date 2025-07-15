#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void thread_1D_indexing(int *a)
{
    int thread_id=threadIdx.x+blockIdx.x*blockDim.x;

    printf("thread index: %d value: %d\n",thread_id,a[thread_id]);
}

int main()
{
    int n=8,nbytes=sizeof(int)*n;
    int h_arr[]{1,2,3,4,5,6,7,8};
    int *d_arr;
    cudaMalloc((void**)&d_arr,nbytes);
    cudaMemcpy(d_arr,h_arr,nbytes,cudaMemcpyHostToDevice);
    dim3 block(n/2);
    dim3 grid(2);
    thread_1D_indexing<<<grid,block>>>(d_arr);
    cudaDeviceSynchronize();
    cudaFree(d_arr);
    cudaDeviceReset();

    return 0;
}