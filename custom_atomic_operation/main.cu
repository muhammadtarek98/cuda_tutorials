#include <bits/stdc++.h>
#include <cuda_runtime.h>
__device__ int custom_atomic_adder(int *ptr,int inc)
{
    int exp=*ptr;
    int val=atomicCAS(ptr,exp,exp+inc);
    while (val!=exp)
    {
        exp=val;
        val=atomicCAS(ptr,exp,exp+inc);
    }
    return val;
}
__global__ void kernel(int *ptr)
{
    custom_atomic_adder(ptr,1);
}
int main()
{
    int val=23;
    int *h_val=new int;
    int *d_val=nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_val),sizeof(int));
    cudaMemcpy(d_val,&val,sizeof(int),cudaMemcpyHostToDevice);
    kernel<<<1,32,0>>>(d_val);
    cudaMemcpy(h_val,d_val,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_val);
    std::cout<<*h_val;
    delete h_val;

    return 0;
}