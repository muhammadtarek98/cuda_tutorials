#include <bits/stdc++.h>
#define TPB 64
__global__ void kernel(float *res,float *a,float *b,int n)
{
    const auto tgid=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ float shared_mem[TPB];
    if (tgid>=n)
    {
        return;
    }
    const auto shared_mem_idx=threadIdx.x;
    shared_mem[shared_mem_idx]=a[shared_mem_idx]*b[shared_mem_idx];
    __syncthreads();
    if (shared_mem_idx==0)
    {
        float block_sum=0.0;
        for (int j=0;j<blockDim.x;j++)
        {
            block_sum+=shared_mem[j];
        }
        atomicAdd(res,block_sum);
    }
}
template<typename T>
__host__ void init(std::unique_ptr<T[]>&ptr,int sz)
{
    for (int i=0;i<sz;i++)
    {
        ptr.get()[i]=rand();
    }
}
int main()
{
    const auto sz=1024;
    const auto  bytes=sz*sizeof(float);
    const dim3 block(TPB);
    const dim3 grid((sz-block.x-1)/block.x);
    std::unique_ptr<float[]> h_a{std::make_unique<float[]>(sz)};
    std::unique_ptr<float[]> h_b{std::make_unique<float[]>(sz)};
    std::unique_ptr<float> h_c{std::make_unique<float>()};

    init<float>(h_a,sz);
    init<float>(h_b,sz);
    float *d_a=nullptr,*d_b=nullptr,*d_c=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_b),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_c),sizeof(float));
    cudaMemset(d_c,0.0,sizeof(float));
    cudaMemcpy(d_a,h_a.get(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.get(),bytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block>>>(d_c,d_a,d_b,sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.get(),d_c,sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b),cudaFree(d_c);
    std::cout<<*h_c.get();

    return 0;
}