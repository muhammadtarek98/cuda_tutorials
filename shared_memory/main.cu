#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#define SHARED_MEM_SIZE 128
const int sz=1<<22,block_size=SHARED_MEM_SIZE,bytes=sizeof(int)*sz;
dim3 blocks(block_size);
dim3 grids((sz/blocks.x)+1);
__global__ void kernel_1(int *a,int *b)
{
    int tid=threadIdx.x;
    int tgid=blockDim.x*blockIdx.x+tid;
    __shared__ int s_arr[SHARED_MEM_SIZE];
    if (tgid<sz)
    {
        s_arr[tid]=a[tgid];
        b[tgid]=s_arr[tid];
    }
}
__global__ void kernel_2(int *a,int *b)
{
    int tid=threadIdx.x;
    int tgid=tid+blockDim.x*blockIdx.x;
    extern __shared__ int s_arr[];
    if (tgid<sz)
    {
        s_arr[tid]=a[tgid];
        b[tgid]=s_arr[tid];
    }
}
void init(std::vector<int>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}
void run_static_shared_mem()
{
    std::vector<int>h_a(sz),h_b(sz);
    int *d_a=nullptr,*d_b=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_b),bytes);
    init(h_a);
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    kernel_1<<<grids,blocks>>>(d_a,d_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b.data(),d_b,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b);
    h_a.clear(),h_b.clear();
}
void run_dynamic_shared_mem()
{
    std::vector<int>h_a(sz),h_b(sz);
    int *d_a=nullptr,*d_b=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_b),bytes);
    init(h_a);
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    kernel_2<<<grids,blocks,SHARED_MEM_SIZE*sizeof(int)>>>(d_a,d_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b.data(),d_b,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b);
    h_a.clear(),h_b.clear();
}
int main()
{
    run_static_shared_mem();
    run_dynamic_shared_mem();
    return 0;
}