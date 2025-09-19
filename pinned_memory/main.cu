#include <bits/stdc++.h>
#include<device_launch_parameters.h>
#include <cuda_runtime.h>

int main(int argc,const char *argv [])
{
    int isize=1<<25;
    auto nbyes=isize* sizeof(float);
    std::vector<float> h_a(isize);std::vector<float>d_a(isize);
    cudaMallocHost(reinterpret_cast<float **>(h_a.data()),nbyes);
    cudaMalloc(reinterpret_cast<float **>(d_a.data()),nbyes);
    for(int i=0;i<isize;++i){
        h_a[i]=5;
    }
    cudaMemcpy(d_a.data(),h_a.data(),nbyes,cudaMemcpyHostToDevice);
    cudaMemcpy(h_a.data(),d_a.data(),nbyes,cudaMemcpyDeviceToHost);
    cudaFree(d_a.data()), cudaFreeHost(h_a.data());

    return 0;
}