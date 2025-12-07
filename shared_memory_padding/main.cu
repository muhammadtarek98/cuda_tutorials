#include <bits/stdc++.h>
#define BDMX 32
#define BDMY 32
#define PAD 1
struct custom_deleter{
    void operator()(int *ptr){
        if(ptr)
            cudaFree(ptr);
    }
};
__global__ void kernel_static_shared_mem(int* out){
    __shared__ int smem[BDMY][BDMX+PAD];
    int tgid=threadIdx.y*blockDim.x+threadIdx.x;
    smem[threadIdx.y][threadIdx.x]=tgid;
    __syncthreads();
    out[tgid]=smem[threadIdx.y][threadIdx.x];
}

__global__ void kernel_dynamic_shared_mem(int* out){
    extern __shared__ int smem[];
    int tgid = threadIdx.y * blockDim.x + threadIdx.x;
    // 2D to 1D mapping with padding
    int smem_idx = (threadIdx.y * blockDim.x + PAD) + threadIdx.x;
    smem[smem_idx] = tgid;
    __syncthreads();

    out[tgid] = smem[smem_idx];
}
void run_static_smem(dim3 &grid,dim3 &block,size_t &bytes){
    cudaSharedMemConfig config;
    cudaDeviceGetSharedMemConfig(&config);
    int *raw_ptr= nullptr;
    cudaMalloc(reinterpret_cast<void **>(&raw_ptr),bytes);
    cudaMemset(raw_ptr,0,bytes);
    std::unique_ptr<int,custom_deleter> d_ptr(raw_ptr);
    auto h_pt=std::make_unique<int[]>(BDMX*BDMX);
    kernel_static_shared_mem<<<grid,block>>>(d_ptr.get());
    cudaDeviceSynchronize();
    cudaMemcpy(h_pt.get(),d_ptr.get(),bytes,cudaMemcpyDeviceToHost);
    for(int i=0;i<BDMY*BDMX;++i){
        std::cout<<h_pt[i]<<"\n";
    }

    cudaDeviceReset();

}
void run_dynamic_smem(dim3 &grid,dim3 &block,size_t &bytes){
    int* d_raw_ptr = nullptr;
    cudaMalloc((void**)&d_raw_ptr, bytes);
    cudaMemset(d_raw_ptr, 0, bytes);
    std::unique_ptr<int, custom_deleter> d_ptr(d_raw_ptr);
    auto h_ptr = std::make_unique<int[]>(BDMX * BDMY);
    kernel_dynamic_shared_mem<<<grid, block, (BDMX + PAD) * BDMY * sizeof(int)>>>(d_ptr.get());
    cudaDeviceSynchronize();
    cudaMemcpy(h_ptr.get(), d_ptr.get(), bytes, cudaMemcpyDeviceToHost);
    for(int i=0; i <BDMY*BDMX; ++i){
        std::cout << h_ptr[i] << "\n";
    }
}
int main() {
    cudaSharedMemConfig config;
    cudaDeviceGetSharedMemConfig(&config);
    dim3 grid(1,1);
    dim3 block(BDMX,BDMY);
    auto  bytes=BDMY*BDMX*sizeof (int);
    run_static_smem(grid,block,bytes);
    run_dynamic_smem(grid,block,bytes);
    return 0;
}
