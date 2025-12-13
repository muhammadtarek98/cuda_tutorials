#include <bits/stdc++.h>
#define arr_sz 128
#define full_mask 0xff
struct customdetr
{
    void operator()(int *ptr)
    {
        cudaFree(ptr);
    }
};
void init(int *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=i%10;
    }
}
__global__ void kernel_lane_32(int *in,int *out)
{
    int x=in[threadIdx.x];
    int y=__shfl_sync(full_mask,x,32);
    out[threadIdx.x]=y;
}

int main()
{
    int bytes=arr_sz*sizeof(int);
    dim3 block(arr_sz);
    dim3 grid(1);
    auto h_in_ptr{std::make_unique<int[]>(arr_sz)};
    auto h_out_ptr{std::make_unique<int[]>(arr_sz)};
    init(h_in_ptr.get(),arr_sz);
    int *d_in_raw=nullptr;
    int *d_out_raw=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in_raw),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out_raw),bytes);
    std::unique_ptr<int,customdetr>d_in_ptr{d_in_raw};
    std::unique_ptr<int,customdetr>d_out_ptr{d_out_raw};
    cudaMemcpy(d_in_ptr.get(),h_in_ptr.get(),bytes,cudaMemcpyHostToDevice);
    kernel_lane_32<<<grid,block>>>(d_in_ptr.get(),d_out_ptr.get());
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_ptr.get(),d_out_ptr.get(),bytes,cudaMemcpyDeviceToHost);
    for (auto i=0;i<arr_sz;++i)
    {
        std::cout<<h_out_ptr.get()[i]<<" ";
    }
    d_in_ptr.release();
    d_out_ptr.release();


    return 0;
}