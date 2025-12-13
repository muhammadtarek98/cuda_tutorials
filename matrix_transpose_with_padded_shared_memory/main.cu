#include <bits/stdc++.h>
#define BDMX 64
#define BDMY 8
#define PAD 2
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

__global__ void kernel(int *in,int *out,int nx,int ny)
{
    __shared__ int smem[BDMY*(2*BDMX+PAD)];
    int ix=2*blockDim.x*blockIdx.x+threadIdx.x;
    int iy=blockDim.y*blockIdx.y+threadIdx.y;
    int in_idx=iy*nx+ix;
    int _1d_idx=threadIdx.y*blockDim.x+threadIdx.x;
    int irow=_1d_idx/blockDim.y;
    int icol=_1d_idx%blockDim.y;
    int out_ix=blockIdx.y*blockDim.y+icol;
    int out_iy=2*blockIdx.x*blockDim.x+irow;
    int out_idx=out_iy*ny+out_ix;
    if (ix<nx&&iy<ny)
    {
        int row_idx=threadIdx.y*(2*blockDim.x+PAD)+threadIdx.x;
        smem[row_idx]=in[in_idx];
        smem[row_idx+BDMX]=in[in_idx+BDMX];
        __syncthreads();
        int col_idx=icol*(2*blockDim.x+PAD)+irow;
        out[out_idx]=smem[col_idx];
        out[out_idx+ny*BDMX]=smem[col_idx+BDMX];
    }
}

int main()
{
    int nx{128},ny{128};
    int sz=nx*ny;
    int bytes=sz*sizeof(int);
    int *d_raw_mat_ptr=nullptr;
    int *d_raw_trans_ptr=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_raw_mat_ptr),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_raw_trans_ptr),bytes);
    auto h_mat_ptr{std::make_unique<int[]>(sz)};
    auto h_trans_ptr{std::make_unique<int[]>(sz)};
    std::unique_ptr<int,customdetr>d_mat{d_raw_mat_ptr};
    std::unique_ptr<int,customdetr>d_trans{d_raw_trans_ptr};
    cudaMemset(d_raw_mat_ptr,0,bytes);
    init(h_mat_ptr.get(),sz);
    cudaMemcpy(d_mat.get(),h_mat_ptr.get(),bytes,cudaMemcpyHostToDevice);
    dim3 block(BDMX,BDMY);
    dim3 grid(nx/block.x,ny/block.y);
    grid.x=grid.x/2;
    kernel<<<grid,block>>>(d_mat.get(),d_trans.get(),nx,ny);
    cudaMemcpy(h_trans_ptr.get(),d_trans.get(),bytes,cudaMemcpyDeviceToHost);
    for (auto i=0;i<nx*ny;++i)
    {
        std::cout<<h_trans_ptr.get()[i]<<" ";
        if ((i+1)%ny==0)
        {
            std::cout<<"\n";
        }
    }

    return 0;
}