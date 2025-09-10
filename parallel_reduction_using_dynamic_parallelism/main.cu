#include<bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void kernel(int *input,int *part,unsigned int size)
{
    auto tid=threadIdx.x;
    int *window=input+blockDim.x*blockIdx.x;
    int *outpart=&part[blockIdx.x];
    if (size==2 && tid==0)
    {
        part[blockIdx.x]=input[0]+input[1];
        return;
    }
    __syncthreads();
    int istride=size>>1;
    if (istride>1 && tid<istride)
    {
        window[tid]+=window[tid+istride];
    }
    __syncthreads();
    if (tid==0)
    {
        kernel<<<1,istride>>>(window,outpart,istride);
        cudaDeviceSynchronize();
    }
    __syncthreads();

}
void init(std::vector<int>&arr)
{
    for (int i=0;i<static_cast<int>(arr.size());++i)
    {
        arr[i]=i%10;
    }
}
void seq_array_accum(int &a, const std::vector<int> &arr)
{
    for (const auto &i : arr)
    {
        a+=i;
    }
}
int main(int argc,const char *argv[])
{
    int size=1<<22,res=0;
    int block_size=512;
    dim3 blocks(block_size);
    dim3 grid((size/block_size));
    const auto input_byte_size=size*sizeof(int);
    const auto part_byte_size=grid.x*sizeof(int);
    std::vector<int> h_input(size);
    std::vector<int>h_part(grid.x,0);
    int *d_input=nullptr;
    int *d_part=nullptr;
    init(h_input);
    cudaMalloc(reinterpret_cast<void**>(&d_input),input_byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_part),part_byte_size);
    cudaMemset(d_part,0,part_byte_size);
    cudaMemcpy(d_input,h_input.data(),input_byte_size,cudaMemcpyHostToDevice);
    kernel<<<grid,blocks>>>(d_input,d_part,block_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
    cudaFree(d_part),cudaFree(d_input);
    cudaDeviceReset();
    seq_array_accum(res,h_part);
    std::cout<<res<<std::endl;
    return 0;
}