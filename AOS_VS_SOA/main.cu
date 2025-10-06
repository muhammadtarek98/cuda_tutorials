#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define  N 1<<22
#define BS 128
struct AOS
{
    float x, y;
    AOS():x(0.0),y(0.0){}
};
struct SOA
{
    float x[N],y[N];
};
__global__ void aos_kernel(AOS *in,AOS *out,const int size){
    auto tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<size)
    {
        AOS temp=in[tgid];
        temp.x+=5,temp.y+=10;
        out[tgid]=temp;
        printf("AOS :threadIdx:%d in_element.x:%f in_element.y:%f out_element.x:%f out_element.y:%f\n",
        tgid,in[tgid].x,in[tgid].y,out[tgid].x,out[tgid].y);
    }
    __syncthreads();
}
__global__ void soa_kernel(SOA *in,SOA *out,const int size)
{
    auto tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<size)
    {
        float tmpx = in->x[static_cast<int>(tgid)];
        float tmpy = in->y[static_cast<int>(tgid)];
        tmpx += 5;
        tmpy += 10;
        out->x[tgid] = tmpx;
        out->y[tgid] = tmpy;
        printf("SOA: threadIdx:%d in_element.x:%f in_element.y:%f out_element.x:%f out_element.y:%f\n",
            tgid,in->x[tgid],in->y[tgid],out->x[tgid],out->y[tgid]);
    }
    __syncthreads();
}
void run_AOS()
{
    auto bytes=sizeof(AOS)*N;
    dim3 blocks(BS);
    dim3 grid(N/BS);
    std::vector<AOS>in(N);
    std::vector<AOS>out(N);
    std::shared_ptr<AOS>d_in,d_out;
    cudaMalloc(reinterpret_cast<void**>(&d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),bytes);
    cudaMemcpy(d_in.get(),in.data(),bytes,cudaMemcpyHostToDevice);
    aos_kernel<<<grid,blocks>>>(d_in.get(),d_out.get(),N);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(),d_out.get(),bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_in.get()),cudaFree(d_out.get());
    cudaDeviceReset();
    in.clear(),out.clear();
}
void run_SOA()
{
    auto bytes=sizeof(SOA);
    dim3 blocks(BS),grid(N/BS);
    std::shared_ptr<SOA>in,out;
    std::shared_ptr<SOA>d_in,d_out;
    cudaMalloc(reinterpret_cast<void**>(&d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),bytes);
    cudaMemcpy(in.get(),d_in.get(),bytes,cudaMemcpyHostToDevice);
    soa_kernel<<<grid,blocks>>>(d_in.get(),d_out.get(),N);
    cudaDeviceSynchronize();
    cudaMemcpy(d_out.get(),out.get(),bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_in.get()),cudaFree(d_out.get());
    in.reset();
    out.reset();
    cudaDeviceReset();
}
int main()
{
    run_AOS();
    run_SOA();
    return 0;
}