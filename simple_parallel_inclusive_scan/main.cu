#include <bits/stdc++.h>
#include<cuda_runtime.h>
__global__ void kernel(int *in,int *out,int sz)
{
    int tid=threadIdx.x;
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<sz)
    {
        for (int i=1;i<=tid;i*=2)
        {
            int curr=in[tgid];
            int prev=in[tgid-1];
            in[tgid]+=curr+prev;
        }
        __syncthreads();

    }
    *out=in[tgid];
}
__host__ void init(int *ptr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        ptr[i]=1;
    }
}
int main()
{
    const auto sz=1<<10;
    const auto blocks=1024;
    auto const bytes=sizeof(int)*sz;
    const dim3 block(blocks);
    const dim3 grid((sz + blocks - 1) / blocks);
    std::unique_ptr<int[]> h_in{std::make_unique<int[]>(sz)};
    std::unique_ptr<int>h_out{std::make_unique<int>()};
    init(h_in.get(),sz);
    int *raw_d_in=nullptr;
    int *raw_d_out=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&raw_d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&raw_d_out),sizeof(int));
    cudaMemcpy(raw_d_in,h_in.get(),bytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block>>>(raw_d_in,raw_d_out,sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.get(),raw_d_out,sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<*h_out;

    return 0;
}