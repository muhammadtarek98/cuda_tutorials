#include <bits/stdc++.h>
#define arr_sz 128
#define full_mask 0xffffffff
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
template<unsigned int iblock_sz>
__global__ void kernel(int *in,int *out,int sz)
{
    __shared__ int smem[arr_sz];
    int tid = threadIdx.x;
    int * i_data = in + blockDim.x * blockIdx.x;

    smem[tid] = i_data[tid];

    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    if (blockDim.x >= 64 && tid < 32)
        smem[tid] += smem[tid + 32];
    __syncthreads();

    int local_sum = smem[tid];

    //unrolling warp
    if (tid < 32)
    {
        local_sum += __shfl_down_sync(full_mask, local_sum, 16);
        local_sum += __shfl_down_sync(full_mask, local_sum, 8);
        local_sum += __shfl_down_sync(full_mask, local_sum, 4);
        local_sum += __shfl_down_sync(full_mask, local_sum, 2);
        local_sum += __shfl_down_sync(full_mask, local_sum, 1);
    }

    if (tid == 0)
    {
        out[blockIdx.x] = local_sum;
    }
}
int main()
{
    int sz = 1 << 25;
    int block_size = arr_sz;
    dim3 block(block_size);
    dim3 grid((sz + block_size - 1) / block_size);
    int bytes=sizeof(int)*grid.x;
    auto h_in_ptr{std::make_unique<int[]>(block_size)};
    auto h_out_ptr{std::make_unique<int>(grid.x)};
    init(h_in_ptr.get(),arr_sz);
    int *d_in_raw=nullptr,*d_out_raw=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in_raw),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out_raw),sizeof(int)*grid.x);
    cudaMemset(d_out_raw,0,sizeof(int)*grid.x);
    std::unique_ptr<int,customdetr> d_in{d_in_raw};
    std::unique_ptr<int,customdetr>d_out{d_out_raw};
    cudaMemcpy(d_in.get(),h_in_ptr.get(),bytes,cudaMemcpyHostToDevice);
    kernel<128><<<grid,block>>>(d_in.get(),d_out.get(),sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_ptr.get(),d_out.get(),sizeof(int)*grid.x,cudaMemcpyDeviceToHost);
    int res=0;
    for (int i=0;i<grid.x;++i)
    {
        res+=h_out_ptr.get()[i];
    }
    std::cout<<res;
    return 0;
}