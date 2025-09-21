#include<bits/stdc++.h>
#include<cuda_runtime.h>
__global__ void kernel (float *a,float *b,float *c,int sz)
{
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid<sz)
    {
        c[tgid]=a[tgid]+b[tgid];
    }
}
void init(float *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=i%10;
    }
}
void seq_array_sum(const float *a, const float *b, float *c,int sz)
{
    for (int i=0;i<sz;++i)
    {
        c[i]+=a[i]+b[i];
    }
}
int main()
{
    int sz=1<<22,block_sz=128;
    dim3 blocks(block_sz);
    dim3 grid((sz+blocks.x-1)/blocks.x);
    auto bytes=sizeof(float)*sz;
    float *a,*b,*c,*ref;
    c=(float*)malloc(bytes);
    cudaMallocManaged(reinterpret_cast<void**>(&a),bytes);
    cudaMallocManaged(reinterpret_cast<void**>(&b),bytes);
    cudaMallocManaged(reinterpret_cast<void**>(&ref),bytes);
    init(a,sz),init(b,sz);
    seq_array_sum(a,b,c,sz);
    kernel<<<grid,blocks>>>(a,b,ref,sz);
    cudaDeviceSynchronize();
    free(c);
    return 0;
}