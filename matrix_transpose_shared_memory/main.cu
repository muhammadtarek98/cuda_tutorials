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
__global__ void kernel(int *mat,int nx,int ny,int *out)
{
    __shared__ int smem[BDMY][BDMX];
    int ix=blockDim.x*blockIdx.x+threadIdx.x;
    int iy=blockDim.y*blockIdx.y+threadIdx.y;
    int in_idx=iy*nx+ix,_1d_idx=threadIdx.y*blockDim.x+threadIdx.x;
    int irow=_1d_idx/blockDim.y,icol=_1d_idx%blockDim.y;
    int out_ix=blockIdx.y*blockDim.y+icol;
    int out_iy=blockDim.x*blockIdx.x+irow;
    int out_idx=out_iy*ny+out_ix;
    if (ix<nx&&iy<ny)
    {
        smem[threadIdx.y][threadIdx.x]=mat[in_idx];
        __syncthreads();
        out[out_idx]=smem[icol][irow];
    }

}
int main()
{

    int nx=1024,ny=1024;
    int block_x=BDMX,block_y=BDMY;
    dim3 block(block_x,block_y);
    dim3 grid(nx/block_x,ny/block_y);
    int sz=nx*ny;
    auto bytes=sizeof(int)*sz;
    int *d_raw_mat_ptr=nullptr;
    int *d_raw_trans_ptr=nullptr;
    auto h_mat_ptr{std::make_unique<int[]>(nx*ny)};
    auto h_trans_ptr{std::make_unique<int[]>(nx*ny)};
    init(h_mat_ptr.get(),sz);
    cudaMalloc(reinterpret_cast<void**>(&d_raw_mat_ptr),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_raw_trans_ptr),bytes);
    std::unique_ptr<int,customdetr> d_mat{d_raw_mat_ptr};
    std::unique_ptr<int,customdetr>d_trans{d_raw_trans_ptr};
    cudaMemset(d_raw_trans_ptr,0,bytes);
    cudaMemcpy(d_mat.get(),h_mat_ptr.get(),bytes,cudaMemcpyHostToDevice);
    kernel<<<grid,block>>>(d_mat.get(),nx,ny,d_trans.get());
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