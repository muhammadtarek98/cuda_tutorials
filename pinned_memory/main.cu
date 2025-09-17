#include <bits/stdc++.h>
#include<device_launch_parameters.h>
#include <cuda_runtime.h>

int main(int argc,const char *argv [])
{
    int isize=1<<25;
    int nbyes=isize* sizeof(float );
    float *h_a,*d_a;
    cudaMallocHost(reinterpret_cast<float **>(&h_a),nbyes);
    cudaMalloc(reinterpret_cast<float **>(&d_a),nbyes);
    for(int i=0;i<isize;++i){
        h_a[i]=5;
    }
    cudaMemcpy(d_a,h_a,nbyes,cudaMemcpyHostToDevice);
    cudaMemcpy(h_a,d_a,nbyes,cudaMemcpyDeviceToHost);
    cudaFree(d_a), cudaFreeHost(h_a);

    return 0;
}