#include<bits/stdc++.h>
#include<device_launch_parameters.h>
#include <cuda_runtime.h>
__global__ void kernel(int *a,int *b,int *c,const int size)
{
    int tgid=threadIdx.x+blockIdx.x*blockDim.x;
    if (tgid<size)
    {
        c[tgid]=a[tgid]+b[tgid];
    }
    __syncthreads();
}
void init(std::vector<int>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}
int main(int argc,const char *argv [])
{
    int size=1<<22,block_size=128;
    dim3 blocksize(block_size);
    dim3 grid((size+block_size-1)/block_size);
    if (argc>1)
    {
        block_size=1<<atoi(argv[1]);
    }
    auto byte_size=size*sizeof(int);
    std::vector<int>a(size);
    std::vector<int>b(size);
    std::vector<int>c(size,0);
    init(a),init(b);

    int *d_a=nullptr,*d_b=nullptr,*d_c=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_b),byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_c),byte_size);
    cudaMemset(d_c,0,byte_size);
    cudaMemcpy(d_a,a.data(),byte_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,b.data(),byte_size,cudaMemcpyHostToDevice);
    kernel<<<grid,blocksize>>>(d_a,d_b,d_c,size);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(),d_c,byte_size,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_c),cudaFree(d_b);
    a.clear();
    b.clear();
    c.clear();

    return 0;
}