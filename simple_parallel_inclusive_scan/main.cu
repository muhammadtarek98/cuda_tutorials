#include <bits/stdc++.h>
#include<cuda_runtime.h>
__global__ void kernel(int *in,int sz)
{
    int tid=threadIdx.x;
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<sz)
    {
        for (int i=1;i<=tid;i*=2)
        {
            in[tgid]+=in[tgid-i];
        }
    }
    __syncthreads();
}
__host__ void init(int *ptr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        ptr[i]=i%10;
    }
}
int main()
{
    const auto sz=1<<10;
    const auto blocks=1024;
    auto const bytes=sizeof(int)*sz;
    const dim3 block(blocks);
    const dim3 grid((sz + blocks - 1) / blocks);
    int *h_in=new int[sz];
    int *h_out=new int;
    init(h_in,sz);
    int *d_in=nullptr;
    int *d_out=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),sizeof(int));
    cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block,0>>>(d_in,sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,&d_in[1],sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<*h_out;
    cudaFree(d_out),cudaFree(d_in);
    delete h_out;delete h_in;

    return 0;
}