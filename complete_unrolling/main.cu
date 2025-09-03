#include<bits/stdc++.h>
#include<device_launch_parameters.h>
#include <cuda_runtime.h>
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
__global__ void kernel(int *input,int *part,int size){
    auto tid=threadIdx.x;
    auto index=tid+blockIdx.x*blockDim.x;
    int *window=input+blockDim.x*blockIdx.x;
    if(blockDim.x>=1024&&tid<512){
    window[tid]+=window[tid+512];
    }
    __syncthreads();
    if(blockDim.x>=512&&tid<256){
    window[tid]+=window[tid+256];
    }
    __syncthreads();
    if(blockDim.x>=256&&tid<128){
        window[tid]+=window[tid+128];
    }
    __syncthreads();
    if(blockDim.x>=128&&tid<64){
        window[tid]+=window[tid+64];
    }
    __syncthreads();
    if(tid<32){
        volatile int *vsmem=window;
        vsmem[tid]+=vsmem[tid+32];
        vsmem[tid]+=vsmem[tid+16];
        vsmem[tid]+=vsmem[tid+8];
        vsmem[tid]+=vsmem[tid+4];
        vsmem[tid]+=vsmem[tid+2];
        vsmem[tid]+=vsmem[tid+1];
    }
    __syncthreads();

    if(tid==0){
    part[blockIdx.x]=window[0];
    }
    __syncthreads();

}
int main(int argc,const char *argv [])
{
    int size=1<<27;
    int block_size=1024;
    int cpu_res=0,gpu_res=0;
    dim3 blocks(block_size);
    dim3 grid(size/block_size);
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
    kernel<<<grid,blocks>>>(d_input,d_part,size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
    seq_array_accum(gpu_res,h_part);
    std::cout<<gpu_res<<" "<<cpu_res<<"\n";
    std::cout<<(gpu_res==cpu_res)<<"\n";
    h_input.clear(),h_part.clear(),cudaFree(d_part),cudaFree(d_input);
    cudaDeviceReset();


    return 0;
}