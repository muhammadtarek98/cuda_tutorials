#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
void seq_array_accum(int &a, const std::vector<int> &arr)
{
    for (const auto &i : arr)
    {
        a+=i;
    }
}
void init(std::vector<int>&arr)
{
    for (int i=0;i<static_cast<int>(arr.size());++i)
    {
        arr[i]=i%10;
    }
}

template<unsigned int iblock_size>
__global__ void kernel(int *input,int *part,int size)
{
    auto tid=threadIdx.x;
    auto index=tid+blockDim.x*blockIdx.x*8;
    int *window=input+blockDim.x*blockIdx.x*8;
    if ((index+7*blockDim.x)<size)
    {
        int a1=input[index+blockDim.x];
        int a2=input[index+2*blockDim.x];
        int a3=input[index+3*blockDim.x];
        int a4=input[index+4*blockDim.x];
        int a5=input[index+5*blockDim.x];
        int a6=input[index+6*blockDim.x];
        int a7=input[index+7*blockDim.x];
        input[index]+=a1+a2+a3+a4+a5+a6+a7;
    }
    __syncthreads();
    if (iblock_size>=1024&&tid<512)
    {
        window[tid]+=window[tid+512];
    }
    __syncthreads();
    if (iblock_size>=512&&tid<256)
    {
        window[tid]+=window[tid+256];
    }
    __syncthreads();
    if (iblock_size>=256&&tid<128)
    {
        window[tid]+=window[tid+128];
    }
    __syncthreads();
    if (iblock_size>=128&&tid<64)
    {
        window[tid]+=window[tid+64];
    }
    __syncthreads();
    if (tid<32)
    {
        volatile int *vsmem=window;
        vsmem[tid]+=vsmem[tid+32];
        vsmem[tid]+=vsmem[tid+16];
        vsmem[tid]+=vsmem[tid+8];
        vsmem[tid]+=vsmem[tid+4];
        vsmem[tid]+=vsmem[tid+2];
        vsmem[tid]+=vsmem[tid+1];
    }
    __syncthreads();
    if (tid==0)
    {
        part[blockIdx.x]=window[0];
    }
    __syncthreads();

}
int main()
{
    constexpr auto size=1<<27;
    constexpr auto block_size=1024;
    auto cpu_res=0,gpu_res=0;
    dim3 blocks(block_size);
    dim3 grid((size/block_size)/8);
    constexpr auto input_byte_size=size*sizeof(int);
    const auto part_byte_size=grid.x*sizeof(int);
    std::vector<int>h_input(size);
    std::vector<int>h_part(grid.x,0);
    int *d_input=nullptr,*d_part=nullptr;
    init(h_input);
    seq_array_accum(cpu_res,h_input);
    cudaMalloc(reinterpret_cast<void**>(&d_input),input_byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_part),part_byte_size);
    cudaMemset(d_part,0,part_byte_size);
    cudaMemcpy(d_input,h_input.data(),input_byte_size,cudaMemcpyHostToDevice);
    switch (block_size)
    {
    case 1024:
        kernel<1024><<<grid,blocks>>>(d_input,d_part,size);
        break;
    case 512:
        kernel<512><<<grid,blocks>>>(d_input,d_part,size);
        break;
    case 256:
        kernel<256><<<grid,blocks>>>(d_input,d_part,size);
        break;
    case 128:
        kernel<128><<<grid,blocks>>>(d_input,d_part,size);
        break;
    case 64:
        kernel<64><<<grid,blocks>>>(d_input,d_part,size);
        break;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
    seq_array_accum(gpu_res,h_part);
    std::cout<<gpu_res<<" "<<cpu_res<<"\n";
    std::cout<<(gpu_res==cpu_res)<<"\n";
    return 0;
}