#include <bits/stdc++.h>
__global__ void kernel(float *a,float *b,float *c,int size,int offset){
int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    int k=tgid+offset;
    if (k<size)
    {
        c[k]=a[tgid]+b[tgid];
    }
}
template<typename T>
void init(std::vector<T>&arr)
{
    for (int i=0;i<(int)arr.size();++i)
    {
        arr[i]=i%10;
    }
}

int main()
{
    int size=1<<25,block_size=128,offset=5;
    auto bytes=sizeof(float)*size;
    dim3 blocks(block_size),grid((size+block_size-1)/block_size);
    std::vector<float> a(size),b(size),c(size,0.0);
    init<float>(a),init<float>(b);
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da),bytes);
    cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&da),bytes);
    cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&dc),bytes);
    cudaMemset(dc,0.0,bytes);
    kernel<<<grid,blocks>>>(da,db,dc,size,offset);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(),dc,bytes,cudaMemcpyDeviceToHost);
    cudaFree(da),cudaFree(db),cudaFree(dc);
    return 0;
}