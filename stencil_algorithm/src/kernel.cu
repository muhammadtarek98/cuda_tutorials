#include "kernel.h"
__global__ void smem(float *in,float *out,int sz,float h)
{
    const int i=threadIdx.x+blockDim.x*blockIdx.x;
    if (i>=sz){return;}
    const int s_i=threadIdx.x+RAD;
    extern __shared__ float s_mem[];
    s_mem[s_i]=in[i];
    if (threadIdx.x<RAD)
    {
        s_mem[s_i-RAD]=in[i-RAD];
        s_mem[s_i+blockDim.x]=in[i+blockDim.x];
    }
    __syncthreads();
    out[i]=(s_mem[s_i-1]-2.f*s_mem[s_i]+s_mem[s_i + 1])/(h*h);
}
void run(float *in,float *out,const int &n,const float &h)
{
    float *d_in=nullptr,*d_out=nullptr;
    auto bytes{n*sizeof(float)};
    cudaMalloc(&d_in,bytes);
    cudaMalloc(&d_out,bytes);
    cudaMemcpy(d_in,in,bytes,cudaMemcpyHostToDevice);
    const auto shared_mem_sz=(TPB+2*RAD)*sizeof(float);
    dim3 grid((n+TPB-1)/TPB);
    dim3 block(TPB);
    smem<<<grid,block,shared_mem_sz>>>(d_in,d_out,n,h);
    cudaMemcpy(out,d_out,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_in),cudaFree(d_out);
}

std::tuple<std::vector<float>, std::vector<float>> init(const float& h, const float& pi, const int& n)
{
    std::vector<float>x(n,0.0),y(n,0.0),result(n,0.0);
    for (int i=0;i<n;++i)
    {
        x[i]=2*pi*i/n;
        y[i]=sinf(x[i]);
    }
    auto arrs{std::make_tuple(y,result)};
    return arrs;
}
