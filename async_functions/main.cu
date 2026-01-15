#include <cuda_runtime.h>
#include <bits/stdc++.h>
struct customdetr
{
    void operator()(int *ptr)
    {
        cudaFree(ptr);
    }
};
__host__ void init(int *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=i%10;
    }
}
__global__ void kernel(int *in,int *out,int sz)
{
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<sz)
    {
        for (int i=0;i<25;++i)
        {
            out[tgid]=in[tgid]+(in[tgid-1]*(tgid%10));
        }
    }
}

int main()
{
    auto sz=1<<18;
    auto bytes=sz*sizeof(int);
    const dim3 block(128);
    const dim3 grid(sz/block.x);
    auto h_in{std::make_unique<int[]>(sz)};
    auto h_out{std::make_unique<int[]>(sz)};
    int *raw_d_in=nullptr;
    int *raw_d_out=nullptr;
    std::unique_ptr<int,customdetr> d_int{raw_d_in};
    std::unique_ptr<int,customdetr> d_out{raw_d_out};
    std::shared_ptr<cudaStream_t> stream{std::make_shared<cudaStream_t>()};
    cudaStreamCreate(stream.get());
    init(h_in.get(),sz);
    cudaMallocHost(reinterpret_cast<void**>(h_in.get()),bytes);
    cudaMallocHost(reinterpret_cast<void**>(h_out.get()),bytes);
    cudaMemcpyAsync(d_int.get(),h_in.get(),bytes,cudaMemcpyHostToDevice,cudaStream_t(stream.get()));
    kernel<<<grid,block,0,cudaStream_t(stream.get())>>>(d_int.get(),d_out.get(),sz);
    cudaMemcpyAsync(d_out.get(),h_out.get(),bytes,cudaMemcpyDeviceToHost,cudaStream_t(stream.get()));
    cudaStreamSynchronize(cudaStream_t(stream.get()));
    cudaStreamDestroy(cudaStream_t(stream.get()));
    d_int.reset(),d_out.reset();




    return 0;
}