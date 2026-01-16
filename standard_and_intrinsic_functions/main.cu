#include <bits/stdc++.h>
#include <cuda_runtime.h>
struct UniquePtrDevice
{
    void operator()(float *ptr)
    {
        if (ptr)
            cudaFree(ptr);
    }
};
__global__ void stand_kernel(float *ptr)
{
    *ptr=powf(*ptr,3.0);
}
__global__ void intrinsic_kernel(float *ptr)
{
    *ptr=__powf(*ptr,3.0);
}
int main()
{
    float a=23.0,stand_res,intrinsic_res;
    float *raw_d_a=nullptr;
    std::unique_ptr<float,UniquePtrDevice>d_a{raw_d_a};
    cudaMalloc(reinterpret_cast<void**>(d_a.get()),sizeof(float));
    cudaMemcpy(d_a.get(),&a,sizeof(float),cudaMemcpyHostToDevice);
    stand_kernel<<<1,1>>>(d_a.get());
    cudaMemcpy(&stand_res,d_a.get(),sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<stand_res<<"\n";
    intrinsic_kernel<<<1,1>>>(d_a.get());
    cudaMemcpy(&intrinsic_res,d_a.get(),sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<intrinsic_res;
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}