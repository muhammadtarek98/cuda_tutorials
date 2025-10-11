#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
const int nx=1024,ny=1024,block_x=128,block_y=8;

void print_matrix(std::vector<int> &matrix)
{
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            printf("%d ",matrix[i+j]);
        }
    }
    printf("\n");
}
void init_matrix(std::vector<int> &a)
{
    int j=0;
    for (int i = 0; i < a.size(); ++i)
    {
        if (i % 5 == 0)
        {
            a[i] = i+j;
        }
        else
        {
            a[i]= i*j+7;
        }
    }
}
__global__ void row_major_kernel(int *a,int *a_trans,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if (ix<nx&&iy<ny)
    {
        a_trans[ix*ny+iy]=a[iy*nx+ix];
    }
    __syncthreads();
}
__global__ void col_major_kernel(int *a,int *a_trans,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    if (ix<nx&&iy<ny)
    {
        a_trans[iy*nx+ix]=a[ix*ny+iy];
    }
    __syncthreads();
}
__global__ void row_major_unrolling(int *a,int *trans,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int ti=iy*nx+ix,to=ix*ny+iy;

    if (ix+3*blockDim.x<nx&&iy<ny)
    {
        trans[to]=a[ti];
        trans[to+blockDim.x*ny]=a[ti+blockDim.x];
        trans[to+2*blockDim.x*ny]=a[ti+2*blockDim.x];
        trans[to+3*blockDim.x*ny]=a[ti+3*blockDim.x];
    }
    __syncthreads();
}
__global__ void col_major_unrolling(int *a,int *trans,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int ti=iy*nx+ix,to=ix*ny+iy;
    if (ix+3*blockDim.x<nx&&iy<ny)
    {
        trans[ti]=a[to];
        trans[ti+blockDim.x]=a[to+blockDim.x*ny];
        trans[ti+2*blockDim.x]=a[to+2*blockDim.x*ny];
        trans[ti+3*blockDim.x]=a[to+3*blockDim.x*ny];
    }
    __syncthreads();

}
__global__ void diagonal_major(int *a,int *trans,int nx,int ny)
{
    int blk_x=blockIdx.x,blk_y=(blockIdx.x+blockIdx.y)%gridDim.x;
    int ix=blockIdx.x*blk_x+threadIdx.x;
    int iy=blockIdx.y*blk_y+threadIdx.y;
    if (ix<nx&&iy<ny)
    {
        trans[ix*ny+iy]=a[iy*nx+ix];
    }
    __syncthreads();
}
void run_diagonal_major_kernel()
{
    printf("run_diagonal_major_kernel\n");
    int sz=ny*nx;
    int bytes=sizeof(int)*sz;
    dim3 blocks(block_x,block_y);
    dim3 grid((nx + block_x - 1) / block_x,(ny + block_y - 1) / block_y);
    std::vector<int> h_a(nx*ny,0);
    std::vector<int>h_a_trans(ny*nx,0);
    int *d_a=nullptr,*d_a_trans=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_a_trans),bytes);
    init_matrix(h_a);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    row_major_kernel<<<grid,blocks>>>(d_a,d_a_trans,nx,ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a_trans.data(),d_a_trans,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_a_trans);
    print_matrix(h_a_trans);
    h_a.clear(),h_a_trans.clear();
    cudaDeviceReset();
}
void run_row_major_kernel()
{
    printf("run_row_major_kernel\n");
    int sz=ny*nx;
    int bytes=sizeof(int)*sz;
    dim3 blocks(block_x,block_y);
    dim3 grid((nx + block_x - 1) / block_x,(ny + block_y - 1) / block_y);
    std::vector<int> h_a(nx*ny,0);
    std::vector<int>h_a_trans(ny*nx,0);
    int *d_a=nullptr,*d_a_trans=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_a_trans),bytes);
    init_matrix(h_a);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    row_major_kernel<<<grid,blocks>>>(d_a,d_a_trans,nx,ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a_trans.data(),d_a_trans,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_a_trans);
    print_matrix(h_a_trans);
    h_a.clear(),h_a_trans.clear();
    cudaDeviceReset();
}
void run_col_major_kernel()
{
    printf("run_col_major_kernel\n");
    int sz=ny*nx;
    int bytes=sizeof(int)*sz;
    dim3 blocks(block_x,block_y);
    dim3 grid((nx + block_x - 1) / block_x,(ny + block_y - 1) / block_y);;
    std::vector<int> h_a(nx*ny,0);
    std::vector<int>h_a_trans(ny*nx,0);
    int *d_a=nullptr,*d_a_trans=nullptr;
    init_matrix(h_a);
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_a_trans),bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    col_major_kernel<<<grid,blocks>>>(d_a,d_a_trans,nx,ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a_trans.data(),d_a_trans,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_a_trans);
    print_matrix(h_a_trans);
    h_a.clear(),h_a_trans.clear();
    cudaDeviceReset();
}

void run_col_major_unroll_kernel()
{
    printf("run_col_major_unroll_kernel\n");
    int sz=ny*nx;
    int bytes=sizeof(int)*sz;
    dim3 blocks(block_x,block_y);
    dim3 grid((nx + block_x - 1) / block_x,(ny + block_y - 1) / block_y);;
    std::vector<int> h_a(nx*ny);
    std::vector<int>h_a_trans(ny*nx);
    int *d_a=nullptr,*d_a_trans=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_a_trans),bytes);
    init_matrix(h_a);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    col_major_unrolling<<<grid,blocks>>>(d_a,d_a_trans,nx,ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a_trans.data(),d_a_trans,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_a_trans);
    print_matrix(h_a_trans);
    h_a.clear(),h_a_trans.clear();
    cudaDeviceReset();
}
void run_row_major_unroll_kernel()
{
    printf("run_row_major_unroll_kernel\n");
    int sz=ny*nx;
    int bytes=sizeof(int)*sz;
    dim3 blocks(block_x,block_y);
    dim3 grid((nx + block_x - 1) / block_x,(ny + block_y - 1) / block_y);;
    std::vector<int> h_a(nx*ny);
    std::vector<int>h_a_trans(nx*ny);
    init_matrix(h_a);
    int *d_a=nullptr,*d_a_trans=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_a_trans),bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    row_major_unrolling<<<grid,blocks>>>(d_a,d_a_trans,nx,ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_a_trans.data(),d_a_trans,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_a_trans);
    print_matrix(h_a_trans);
    h_a.clear(),h_a_trans.clear();
    cudaDeviceReset();
}

int main()
{
    run_row_major_kernel();
    run_col_major_kernel();
    run_col_major_unroll_kernel();
    run_row_major_unroll_kernel();
    run_diagonal_major_kernel();
    return 0;
}