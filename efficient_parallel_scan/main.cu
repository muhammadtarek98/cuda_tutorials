#include <bits/stdc++.h>
#include<cuda_runtime.h>
__global__ void kernel(int *in,int *out,int sz)
{
    int tid=threadIdx.x;
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<sz)
    {
        for (int i=1;i<=blockDim.x;i*=2)
        {
            int idx=(tid+1)*2*i-1;
            if (idx<blockDim.x)
            {
                in[idx]+=in[idx-i];
            }
            __syncthreads();
        }
    }
    if (tid==0)
    {
        in[blockDim.x-1]=0;
    }
    int temp=0;
    for (int i=blockDim.x/2;i>0;i/=2)
    {
        int idx=(tid+1)*2*i-1;
        if (idx<blockDim.x)
        {
            temp=in[idx-i];
            in[idx-i]=in[idx];
            in[idx]+=temp;
        }
        __syncthreads();
    }
        *out=in[blockDim.x-1];
}
__host__ void init(int *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=(i+1)%10;
    }
}
int main()
{
    const auto sz=1<<10;
    const auto bytes=sz*sizeof(int);
    const dim3 block(512);
    const dim3 grid((sz + block.x - 1) / block.x);
    int *h_in=new int[sz];
    init(h_in,sz);
    int *h_out=new int;
    int *d_in=nullptr;
    int *d_out=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),sizeof(int));
    cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block,0>>>(d_in,d_out,sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_in),cudaFree(d_out);
    std::cout<<*h_out;
    delete h_out;delete h_in;
    return 0;
}