#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bits/stdc++.h>
__global__ void kernel(float *a,float *b,float *c,int size,int offset)
{
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    int window=tgid+offset;
    if (window<size)
    {
        c[tgid]=a[window]+b[window];
    }
    __syncthreads();
}
template<typename  T>
void init(std::vector<T>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}
int main()
{
    /*
     * to enable l1 cache while compiling by:
        nvcc -xptxa -dlcm=ca -o -main.out main.cu
    * to disable l1 cache while compiling by:
        nvcc -xptxa -dlcm=cg -o -main.out main.cu
    then profile that with nvfpro
     */
    int size=1<<25,offset=9,block_size=128;
    dim3 blocks(block_size),grids((size+block_size-1)/block_size);
    auto bytes=sizeof(float)*size;
    std::vector<float>a(size),b(size),c(size,0.0);
    init<float>(a),init<float>(b);
    float *d_a=nullptr,*d_b=nullptr,*d_c=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMemcpy(a.data(),d_a,bytes,cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&d_b),bytes);
    cudaMemcpy(b.data(),d_b,bytes,cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d_c),bytes);
    cudaMemset(reinterpret_cast<void**>(&d_c),0.0,bytes);
    kernel<<<grids,blocks>>>(d_a,d_b,d_c,size,offset);
    cudaDeviceSynchronize();
    cudaMemcpy(d_c,c.data(),bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_c),cudaFree(d_a),cudaFree(d_b);

    return 0;
}