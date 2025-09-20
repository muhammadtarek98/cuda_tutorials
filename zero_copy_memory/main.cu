#include <bits/stdc++.h>
#include <cuda_runtime.h>
void init(std::vector<int>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}
void init(int *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=i%10;
    }
}
__global__ void kernel(int *a,int *b,int *c,const int n)
{
    auto gtid=blockDim.x*blockIdx.x+threadIdx.x;
    if (gtid<n)
    {
        c[gtid]=a[gtid]+b[gtid];
    }
}
void run_normal_kernel(const int &sz,const int &bytes,dim3 &grid,dim3 &block)
{
    std::vector<int>h_a(sz);
    std::vector<int>h_b(sz);
    std::vector<int>h_c(sz,0);
    init(h_a),init(h_b);
    int *d_a=nullptr,*d_b=nullptr,*d_c=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_b),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_c),bytes);
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block>>>(d_a,d_b,d_c,sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b),cudaFree(d_c);
    h_a.clear(),h_b.clear(),h_c.clear();
}
void run_zero_cpy_memory(const int &sz,const int &bytes,dim3 &grid,dim3 &block)
{
    int *h_a=nullptr;
    int *h_b=nullptr;
    int *h_c=nullptr;
    int *d_a,*d_b,*d_c;
    cudaHostAlloc(reinterpret_cast<void**>(&h_a),bytes,cudaHostAllocMapped);
    cudaHostAlloc(reinterpret_cast<void**>(&h_b),bytes,cudaHostAllocMapped);
    cudaHostAlloc(reinterpret_cast<void**>(&h_c),bytes,cudaHostAllocMapped);
    init(h_a,sz),init(h_b,sz);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_a),(void*)h_a,0);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_b),(void*)h_b,0);
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_c),(void*)h_c,0);
    kernel<<<grid,block>>>(d_a,d_b,d_c,sz);
    cudaDeviceSynchronize();
    cudaFreeHost(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

}
int main(int argc,const char *argv [])
{
    int dev_idx=0;
    cudaDeviceProp dev_pro;
    cudaGetDeviceProperties(&dev_pro,dev_idx);
    if (!dev_pro.canMapHostMemory)
    {
        std::cout<<"doesn't provide zero CPY"<<std::endl;
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    int sz=1<<22;
    auto bytes=sz*sizeof(sz);
    dim3 block(512);
    dim3 grid((sz+block.x-1)/block.x);
    run_normal_kernel(sz,bytes,grid,block);
    run_zero_cpy_memory(sz,bytes,grid,block);
    return 0;
}