#include <bits/stdc++.h>
#include <cuda_runtime.h>
struct CustomDevicePTRDeleter
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

__global__ void kernel(int *a,int *b,int *c,int sz)
{
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<sz)
    {
        c[tgid]=a[tgid]+b[tgid];
    }
}
__host__ void print(int*ptr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        std::cout<<ptr[i]<<"\n";

    }

}
int main()
{

    const auto sz{1<<25}, block{128},num_streams{8};
    const auto bytes{sz*sizeof(int)},bytes_per_stream{bytes/num_streams};
    const auto elements_per_stream{sz/num_streams};
    auto offest=0;
    const dim3 blocks(block);
    const dim3 grid((elements_per_stream + block + 1) );
    std::array<cudaStream_t,num_streams> streams;
    for (auto &stream:streams)
    {
        cudaStreamCreate(&stream);
    }
    std::vector<int>h_a(sz);
    std::vector<int>h_b(sz);
    std::vector<int>h_c(sz);
    int *d_a=nullptr;
    int *d_b=nullptr;
    int *d_c=nullptr;
    cudaMallocHost(reinterpret_cast<void**>(h_a.data()),bytes);
    cudaMallocHost(reinterpret_cast<void**>(h_b.data()),bytes);
    cudaMallocHost(reinterpret_cast<void**>(h_c.data()),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_b),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_c),bytes);

    init(h_a.data(),sz);
    init(h_b.data(),sz);

    for (int i=0;i<num_streams;i++)
    {
        offest=i*elements_per_stream;
        cudaMemcpyAsync(&d_a[offest],h_a.data()+offest,bytes_per_stream,cudaMemcpyHostToDevice,streams[i]);
        cudaMemcpyAsync(&d_b[offest],h_b.data()+offest, bytes_per_stream,cudaMemcpyHostToDevice,streams[i]);
        kernel<<<grid,blocks,0,streams[i]>>>(&d_a[offest],&d_b[offest],&d_c[offest],sz);
        cudaMemcpyAsync(h_c.data()+offest,&d_c[offest], bytes_per_stream,cudaMemcpyDeviceToHost,streams[i]);

    }
    for (auto &stream:streams)
    {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize();
    print(h_c.data(),sz);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a.data());
    cudaFreeHost(h_b.data());
    cudaFreeHost(h_c.data());

    cudaDeviceReset();

    return 0;
}