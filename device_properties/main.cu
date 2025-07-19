#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
int main()
{
    int device_counter=0;
    cudaGetDeviceCount(&device_counter);
    if (device_counter==0)
    {
        std::cout<<"No cuda Device support"<<std::endl;
    }
    int device_idx=0;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props,device_idx);
    std::cout<<  props.multiProcessorCount<<std::endl;
    std::cout<<  props.maxThreadsPerBlock<<std::endl;
    std::cout<< props.totalGlobalMem/1024.0<<std::endl ;
    std::cout<< props.sharedMemPerBlock/1024.0<<std::endl;
    std::cout<< props.maxThreadsPerMultiProcessor<<std::endl;
    std::cout<<props.maxGridSize<<std::endl;
    std::cout<< props.maxBlocksPerMultiProcessor<<std::endl;

    return 0;
}