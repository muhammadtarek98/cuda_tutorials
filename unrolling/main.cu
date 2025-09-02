#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
void init(std::vector<int>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}
__global__ void unrolling_2_blocks(int *input, int *part,int size)
{
    auto tid=threadIdx.x;
    auto block_offset=blockIdx.x*blockDim.x*2;
    auto index=block_offset+tid;
    int *window=input+block_offset;
    if ((index+blockDim.x)<size)
    {
        input[index]+=input[index+blockDim.x];
    }
    __syncthreads();
    for (int offset=blockDim.x/2;offset>0;offset=offset/2)
    {
        if (tid<offset)
        {
            window[tid]+=window[tid+offset];
        }
        __syncthreads();
    }
    if (tid==0)
    {
        part[blockIdx.x]=window[0];
    }
    __syncthreads();

}
__global__ void unrolling_4_blocks(int *input, int *part,int size)
{
    auto tid=threadIdx.x;
    auto block_offset=blockIdx.x*blockDim.x*4;
    auto index=block_offset+tid;
    int *window=input+block_offset;
    if ((index+3*blockDim.x)<size)
    {
        int a1=input[index];
        int a2=input[index+blockDim.x];
        int a3=input[index+2*blockDim.x];
        int a4=input[index+3*blockDim.x];

        input[index]=a1+a2+a3+a4;
    }
    __syncthreads();
    for (auto offset=blockDim.x/2;offset>0;offset/=2)
    {
        if (tid<offset)
        {
            window[tid]+=window[tid+offset];
        }
        __syncthreads();
    }
    if (tid==0)
    {
        part[blockIdx.x]=window[0];
    }

}
void seq_array_accum(int &a, const std::vector<int> &arr)
{
    for (const auto &i : arr)
    {
        a+=i;
    }
}
void run_unrolling_2blocks_kernel()
{
    int size=1<<27;
    int block_size=128;
    int cpu_res=0,gpu_res=0;
    dim3 blocks(block_size);
    dim3 grid((size/block_size)/2);
    const auto input_byte_size=size*sizeof(int);
    const auto part_byte_size=grid.x*sizeof(int);
    std::vector<int> h_input(size);
    std::vector<int>h_part(grid.x,0);
    int *d_input=nullptr;
    int *d_part=nullptr;
    init(h_input);
    seq_array_accum(cpu_res,h_input);
    cudaMalloc(reinterpret_cast<void**>(&d_input),input_byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_part),part_byte_size);
    cudaMemset(d_part,0,part_byte_size);
    cudaMemcpy(d_input,h_input.data(),input_byte_size,cudaMemcpyHostToDevice);
    unrolling_2_blocks<<<grid,blocks>>>(d_input,d_part,size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
    seq_array_accum(gpu_res,h_part);
    std::cout<<gpu_res<<" "<<cpu_res<<"\n";
    std::cout<<(gpu_res==cpu_res)<<"\n";
    h_input.clear(),h_part.clear(),cudaFree(d_part),cudaFree(d_input);
    cudaDeviceReset();

}
void run_unrolling_4blocks_kernel()
{
    int size=1<<27;
    int block_size=128;
    int cpu_res=0,gpu_res=0;
    dim3 blocks(block_size);
    dim3 grid((size/block_size)/4);
    const auto input_byte_size=size*sizeof(int);
    const auto part_byte_size=grid.x*sizeof(int);
    std::vector<int> h_input(size);
    std::vector<int>h_part(grid.x,0);
    int *d_input=nullptr;
    int *d_part=nullptr;
    init(h_input);
    seq_array_accum(cpu_res,h_input);
    cudaMalloc(reinterpret_cast<void**>(&d_input),input_byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_part),part_byte_size);
    cudaMemset(d_part,0,part_byte_size);
    cudaMemcpy(d_input,h_input.data(),input_byte_size,cudaMemcpyHostToDevice);
    unrolling_4_blocks<<<grid,blocks>>>(d_input,d_part,size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
    seq_array_accum(gpu_res,h_part);
    std::cout<<gpu_res<<" "<<cpu_res<<"\n";
    std::cout<<(gpu_res==cpu_res)<<"\n";
    h_input.clear(),h_part.clear(),cudaFree(d_part),cudaFree(d_input);
    cudaDeviceReset();
}
int main()
{
    run_unrolling_2blocks_kernel();
    run_unrolling_4blocks_kernel();
    return 0;
}