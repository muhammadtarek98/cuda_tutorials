#include <bits/stdc++.h>
#define BLOCK_SIZE 1024
void init(int *arr,int sz)
{
    for (int i=0;i<sz;++i)
    {
        arr[i]=i%10;
    }
}
struct custom_deter
{
    void operator()(int *ptr)const
    {
        cudaFree(ptr);
    }
};
template<unsigned int iblocksize>
__global__ void kernel(int *in, int *temp, int sz) {
    unsigned int tid = threadIdx.x;
    unsigned int tgid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int smem[BLOCK_SIZE];

    // Load data into shared memory with bounds checking
    if (tgid < sz) {
        smem[tid] = in[tgid];
    } else {
        smem[tid] = 0;  // Pad with zeros for out-of-bounds threads
    }

    __syncthreads();

    // Reduction in shared memory (using tid, not tgid!)
    if (iblocksize >= 1024 && tid < 512) {
        smem[tid] += smem[tid + 512];
    }
    __syncthreads();

    if (iblocksize >= 512 && tid < 256) {
        smem[tid] += smem[tid + 256];
    }
    __syncthreads();

    if (iblocksize >= 256 && tid < 128) {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();

    if (iblocksize >= 128 && tid < 64) {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();
    if (tid < 32) {
        volatile int *vsmem = smem;  // volatile for warp-synchronous programming
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    __syncthreads();
    if (tid == 0) {
        temp[blockIdx.x] = smem[0];
    }
}
int main() {
    int sz=1<<22,res=0;
    auto bytes=sizeof(int)*sz;
    dim3 blocksize(BLOCK_SIZE);dim3 grid((sz+BLOCK_SIZE-1)/BLOCK_SIZE);
    int *d_raw_ptr = nullptr;
    int *d_temp_raw_ptr = nullptr;
    auto h_ptr=std::make_unique<int[]>(sz);
    auto h_temp_ptr=std::make_unique<int[]>(grid.x);
    init(h_ptr.get(),sz);
    cudaMalloc(reinterpret_cast<void**>(&d_raw_ptr),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_temp_raw_ptr),grid.x*sizeof(int));
    std::unique_ptr<int,custom_deter> d_ptr(d_raw_ptr);
    std::unique_ptr<int,custom_deter> d_temp_ptr(d_temp_raw_ptr);
    cudaMemset(d_temp_ptr.get(), 0, grid.x * sizeof(int));
    cudaMemcpy(d_ptr.get(),h_ptr.get(),bytes,cudaMemcpyHostToDevice);
    kernel<512><<<grid,blocksize>>>(d_ptr.get(),d_temp_ptr.get(),sz);
    cudaDeviceSynchronize();
    cudaMemcpy(h_temp_ptr.get(),d_temp_ptr.get(),grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i=0;i<grid.x;++i)
    {
        res+=h_temp_ptr.get()[i];
    }
    std::cout<<res;
    return 0;
}
