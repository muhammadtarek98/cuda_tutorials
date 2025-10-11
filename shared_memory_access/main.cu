#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define BDMX 32
#define BDMY 32

const int bytes=sizeof(int)*BDMX*BDMY,nx=BDMX,ny=BDMY;
dim3 grid (1,1),blocks(nx,ny);
__global__ void kernel_1(int *out){
    __shared__ int sm[BDMY][BDMX];
    int tidx=threadIdx.x,tidy=threadIdx.y;
    int tgid=tidx+tidy*blockDim.x;
    sm[tidy][tidx]=tgid;
    __syncthreads();
    out[tgid]=sm[tidx][tidy];
}
__global__ void kernel_2(int *out){
    __shared__ int sm[BDMY][BDMX];
    int tidx=threadIdx.x,tidy=threadIdx.y;
    int tgid=tidx+tidy*blockDim.x;
    sm[tidx][tidy]=tgid;
    __syncthreads();
    out[tgid]=sm[tidy][tidx];
}
__global__ void kernel_3(int *out){
    __shared__ int sm[BDMY][BDMX];
    int tidx=threadIdx.x,tidy=threadIdx.y;
    int tgid=tidx+tidy*blockDim.x;
    sm[tidy][tidx]=tgid;
    __syncthreads();
    out[tgid]=sm[tidy][tidx];
}
void run_read_row_store_col_kernel(cudaSharedMemConfig sm_config)
{
    cudaDeviceSetSharedMemConfig(sm_config);
    std::vector<int>h_a(nx*ny);
    int *d_a= nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_a),bytes);
    cudaMemset(d_a,0,bytes);
    kernel_1<<<grid,blocks>>>(d_a);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a.data(),d_a,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    h_a.clear();
    cudaDeviceReset();
}
void run_read_col_store_row_kernel(cudaSharedMemConfig sm_config)
{
    cudaDeviceSetSharedMemConfig(sm_config);
    std::vector<int>h_a(nx*ny);
    int *d_a= nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_a),bytes);
    cudaMemset(d_a,0,bytes);
    kernel_2<<<grid,blocks>>>(d_a);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a.data(),d_a,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    h_a.clear();
    cudaDeviceReset();
}
void run_read_row_store_row_kernel(cudaSharedMemConfig sm_config)
{
    cudaDeviceSetSharedMemConfig(sm_config);
    std::vector<int>h_a(nx*ny);
    int *d_a= nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_a),bytes);
    cudaMemset(d_a,0,bytes);
    kernel_3<<<grid,blocks>>>(d_a);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a.data(),d_a,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    h_a.clear();
    cudaDeviceReset();
}
int main() {
    cudaSharedMemConfig sm_32(cudaSharedMemBankSizeFourByte);
    cudaSharedMemConfig sm_64(cudaSharedMemBankSizeEightByte);
    run_read_col_store_row_kernel(sm_32);
    run_read_row_store_col_kernel(sm_32);
    run_read_row_store_row_kernel(sm_32);
    run_read_col_store_row_kernel(sm_64);
    run_read_row_store_col_kernel(sm_64);
    run_read_row_store_row_kernel(sm_64);

    return 0;
}
