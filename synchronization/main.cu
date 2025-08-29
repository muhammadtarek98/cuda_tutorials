#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
bool compare_results(const int &cpu_res,const int &gpu_res)
{
        if (gpu_res!=cpu_res)
        {
            return false;
        }
    return true;
}
void init(std::vector<int>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}
__global__ void arr_accum(int *arr,int *chunk,int size)
{
    auto tid=threadIdx.x;
    auto tgid=blockDim.x*blockIdx.x+tid;
    if (tgid>size)
    {
        return;
    }
    for (int offset=1;offset<=blockDim.x/2;offset*=2)
    {
         if (tid%(2*offset)==0)
         {
             arr[tgid]+=arr[tgid+offset];
         }
        __syncthreads();
    }
    if (tid==0)
    {
        chunk[blockIdx.x]=arr[tgid];
    }

}
void seq_array_accum(int &a, const std::vector<int> &arr)
{
    for (const auto &i : arr)
    {
        a+=i;
    }
}

int main(){
    constexpr int size=1<<27;
    constexpr int block_size=128;
    int cpu_res=0,gpu_res=0;
    dim3 block(block_size);
    dim3 grid(size/block.x);
    constexpr auto input_byte_size=size*sizeof(int);
    const auto part_byte_size=grid.x*sizeof(int);
    std::vector<int> h_input(size);
    std::vector<int>h_part(grid.x);
    int *d_input=nullptr;
    int *d_part=nullptr;
    init(h_input);
    seq_array_accum(cpu_res,h_input);
    cudaMalloc(reinterpret_cast<void**>(&d_input),input_byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_part),part_byte_size);
    cudaMemset(d_part,0,part_byte_size);
    cudaMemcpy(d_input,h_input.data(),input_byte_size,cudaMemcpyHostToDevice);
    arr_accum<<<grid,block>>>(d_input,d_part,size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
    seq_array_accum(gpu_res,h_part);
    std::cout<<compare_results(cpu_res,gpu_res)<<std::endl;
    std::cout<<cpu_res<<" "<<gpu_res;
    cudaFree(d_input);
    cudaFree(d_part);
    h_input.clear();
    h_part.clear();
    cudaDeviceReset();
    return 0;
}