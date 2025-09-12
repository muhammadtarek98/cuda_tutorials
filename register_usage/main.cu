#include<bits/stdc++.h>
#include<device_launch_parameters.h>
#include <cuda_runtime.h>
__global__ void  kernel(int *res,int size){
    auto tgid=threadIdx.x+blockDim.x*blockIdx.x;
    int x1=21,x2=45,x3=54;
    int x4=x1+x2+x3;
    if(tgid<size){
        res[tgid]=x4;
    }
    __syncthreads();

}
int main(int argc,const char *argv [])
{
    /*
     compile with nvcc --ptxas-options=-v -o main.out main.cu
     */
    int block_size=128;
    int size=1<<22;
     dim3 block(block_size);
    int byte_size=sizeof (int)*size;
    int *h_res=(int*)(malloc(byte_size));
    int *d_res=(int*)(malloc(byte_size));
    cudaMalloc(reinterpret_cast<void **>(&d_res),byte_size);
    cudaMemset(d_res,0,byte_size);
     dim3 grid((size+block.x-1)/block.x);
    kernel<<<grid,block>>>(d_res,size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res,d_res,byte_size,cudaMemcpyDeviceToHost);
    cudaFree(h_res);
    std::cout<<*h_res;
    delete h_res;

    return 0;
}