#include <bits/stdc++.h>
#include <cuda_runtime.h>
__global__ void normal_incremental(int *ptr)
{
    int temp=*ptr;
    temp+=1;
    *ptr=temp;
}
__global__ void atomic_incremental(int *ptr)
{
    atomicAdd(ptr,1);
}
int main()
{
    int v=0;
    int *d_v_n=nullptr;
    int *d_v_a=nullptr;
    int *h_v_n=new int;
    int *h_v_a=new int;
    cudaMalloc(reinterpret_cast<void**>(&d_v_n),sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_v_a),sizeof(int));
    cudaMemcpy(d_v_n,&v,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_a,&v,sizeof(int),cudaMemcpyHostToDevice);
    normal_incremental<<<1,32,0>>>(d_v_n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_v_n,d_v_n,sizeof(int),cudaMemcpyDeviceToHost);
    atomic_incremental<<<1,32,0>>>(d_v_a);
    cudaDeviceSynchronize();
    cudaMemcpy(h_v_a,d_v_a,sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<*h_v_a<<"\n";
    std::cout<<*h_v_n<<"\n";
    cudaFree(d_v_n),cudaFree(d_v_a);
    delete h_v_a;
    delete h_v_n;
    return 0;
}