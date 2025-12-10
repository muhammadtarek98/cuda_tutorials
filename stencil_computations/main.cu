#include <bits/stdc++.h>
#define c0 1
#define c1 2
#define c2 3
#define c3 4
#define c4 5
#define radius 4
#define BDMX 128
struct customdetr
{
    void operator()(int *ptr)
    {
        cudaFree(ptr);
    }
};
void init(int *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=i%10;
    }
}
__constant__ int coef[9];
void set_constant_mem()
{
    const int h_coef[]{c0,c1,c2,c3,c4,c3,c2,c1,c0};
    cudaMemcpyToSymbol(coef,h_coef,sizeof(int)*9);
}
__global__ void kernel(int *in,int *out,int sz)
{
    __shared__ int smem[BDMX+2*radius];
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    int bid=blockIdx.x;
    int num_blocks=gridDim.x;
    int val=0;
    if (tgid<sz)
    {
        auto sidx=threadIdx.x+radius;
        smem[sidx]=in[tgid];
        if (bid!=0&&bid!=(num_blocks-1))
        {
            if (threadIdx.x<radius)
            {
                smem[sidx-radius]=in[tgid-radius];
            }
        }
        else if (bid==0)
        {
            if (threadIdx.x<radius)
            {
                smem[sidx-radius]=in[tgid-radius];
                smem[sidx+radius]=in[tgid+BDMX];
            }
        }
        else
        {
            if (threadIdx.x<radius)
            {
                smem[sidx-radius]=in[tgid-radius];
                smem[sidx+BDMX]=0;
            }
        }
        __syncthreads();

        val += smem[sidx - 4] * coef[0];
        val += smem[sidx - 3] * coef[1];
        val += smem[sidx - 2] * coef[2];
        val += smem[sidx - 1] * coef[3];
        val += smem[sidx - 0] * coef[4];
        val += smem[sidx + 1] * coef[5];
        val += smem[sidx + 2] * coef[6];
        val += smem[sidx + 3] * coef[7];
        val += smem[sidx + 4] * coef[8];
        out[tgid] = val;


    }
}
int main()
{
    auto sz=1<<22;
    auto bytes=sizeof(int)*sz;
   dim3 block(BDMX);
    dim3 grid((sz+BDMX-1)/BDMX);
    int *d_in_ptr=nullptr;
    int *d_out_ptr=nullptr;
    auto h_in_ptr{std::make_unique<int[]>(sz)};
    auto h_out_ptr{std::make_unique<int[]>(sz)};
    init(h_in_ptr.get(),sz);
    cudaMalloc(reinterpret_cast<void**>(&d_in_ptr),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out_ptr),bytes);
    std::unique_ptr<int,customdetr> d_in{d_in_ptr};
    std::unique_ptr<int,customdetr>d_out{d_out_ptr};
    cudaMemset(d_out_ptr,0,bytes);
    cudaMemcpy(d_in.get(),h_in_ptr.get(),bytes,cudaMemcpyHostToDevice);
    set_constant_mem();
    kernel<<<grid,block>>>(d_in.get(),d_out.get(),sz);
    cudaMemcpy(h_out_ptr.get(),d_out.get(),bytes,cudaMemcpyDeviceToHost);
    for (auto i=0;i<sz;++i)
    {
        std::cout<<h_out_ptr.get()[i]<<" ";
    }
    return 0;
}