#include <bits/stdc++.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
__global__ void kernel()
{
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    printf("thread global idx:%d thread idx:%d\n",tgid,threadIdx.x);
}
struct StreamPtrDeleter {
    void operator()(cudaStream_t* ptr) {
        if (ptr && *ptr) {
            cudaStreamDestroy(*ptr);
        }
        delete ptr;  // Delete the allocated pointer itself
    }
};

int main()
{
    int dev_idx=0;
    std::shared_ptr<cudaDeviceProp> dev_pro{std::make_shared<cudaDeviceProp>()};
    cudaGetDeviceProperties(dev_pro.get(),dev_idx);
    if (dev_pro->concurrentKernels==0)
    {
        std::cout<<"device doesn't support concurrect kernels\n";
    }
    std::unique_ptr<cudaStream_t,StreamPtrDeleter> str1{std::move(new cudaStream_t(nullptr))};
    std::unique_ptr<cudaStream_t,StreamPtrDeleter> str2{std::move(new cudaStream_t(nullptr))};
    std::unique_ptr<cudaStream_t,StreamPtrDeleter> str3{std::move(new cudaStream_t(nullptr))};
    cudaStreamCreate(str1.get());
    cudaStreamCreate(str2.get());
    cudaStreamCreate(str3.get());
    kernel<<<32,1,0,*str1>>>();
    cudaDeviceSynchronize();
    kernel<<<32,1,0,*str2>>>();
    cudaDeviceSynchronize();
    kernel<<<32,1,0,*str3>>>();
    cudaDeviceSynchronize();
    cudaStreamSynchronize(*str1);
    cudaStreamSynchronize(*str2);
    cudaStreamSynchronize(*str3);
    cudaDeviceReset();

    return 0;
}