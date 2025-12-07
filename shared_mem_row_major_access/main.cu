#include <bits/stdc++.h>
#define BDMX 32
#define BDMY 32
__global__ void kernel(int *out){
    extern __shared__ int tile[];
    auto row_idx=threadIdx.y*blockDim.x+threadIdx.x;
    auto col_idx=threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx]=row_idx;
    __syncthreads();
    out[row_idx]=tile[col_idx];
}

int main() {
    dim3 block(BDMX,BDMY),grid(1,1);
    auto bytes=BDMY*BDMX*sizeof(int);
    cudaSharedMemConfig smemconfig;
    cudaDeviceGetSharedMemConfig(&smemconfig);
    int *a= nullptr;
    std::vector<int>b(BDMY*BDMY);
    cudaMalloc(reinterpret_cast<void **>(&a),bytes);
    cudaMemset(reinterpret_cast<void **>(&a),0,bytes);
    kernel<<<grid,block,bytes>>>(a);
    cudaDeviceSynchronize();
    cudaMemcpy(b.data(),a,bytes,cudaMemcpyDeviceToHost);
    cudaFree(a);
    for (const auto i:b){
        std::cout<<i<<"\n";
    }
    cudaDeviceReset();
    b.clear();

    return 0;
}
