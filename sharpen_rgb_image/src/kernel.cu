#include "kernel.cuh"

__device__ unsigned char clip(int n)
{
    if (n>=255)
    {
        return 255;
    }
    else if (n<=0)
    {
        return 0;
    }
    return n ;
}
__device__ int clip_idx(int idx,int idx_mx)
{
    if (idx>=idx_mx-1)
    {
        return idx_mx-1;
    }
    else if (idx<=0)
    {
        return 0;
    }
    return idx ;
}
__device__ int flatten(int col,int row,int w,int h)
{
    int col_idx=clip_idx(col,w);
    int row_idx=clip_idx(row,h);
    return col_idx+row_idx*w;

}
__device__ void apply_sharpen(int global_col,int global_row,int g_img_idx)
{
    //todo implement the sharpen for clean code in all three kennels
}
__global__ void normal_sharpen_kernel(uchar4 *out,const uchar4 *in,const float *filter,int w,int h)
{
    int col=threadIdx.x+blockDim.x*blockIdx.x;
    int row=threadIdx.y+blockDim.y*blockIdx.y;
    int g_img_idx=flatten(col,row,w,h);
    int filter_sz=2*RAD+1;
    float channels[3]={0.f,0.f,0.f};
    if ((col>=w)||(row>=h))
    {
        return;
    }
    for (int row_dis=-RAD;row_dis<=RAD;row_dis++)
    {
        for (int col_dis=-RAD;col_dis<=RAD;col_dis++)
        {
            int img_idx=flatten(col+col_dis,row+row_dis,w,h);
            int filter_idx=flatten(RAD+col_dis,RAD+row_dis,filter_sz,filter_sz);
            uchar4 color=in[img_idx];
            float filter_val=filter[filter_idx];
            channels[0]+=filter_val*color.x;
            channels[1]+=filter_val*color.y;
            channels[2]+=filter_val*color.z;
        }
    }

    out[g_img_idx].x=clip(int(channels[0]));
    out[g_img_idx].y=clip(int(channels[1]));
    out[g_img_idx].z=clip(int(channels[2]));
}
__global__ void single_shared_mem_sharpen_kernel(uchar4 *out,const uchar4 *in,const float *filter,int w,int h)
{
    extern __shared__ uchar4 shared_mem [];
    auto tid_x=threadIdx.x,tid_y=threadIdx.y;
    auto col=tid_x+blockDim.x*blockIdx.x, row=tid_y+blockDim.y*blockIdx.y;
    if ((col>=w)||(row>=h))
    {
        return;
    }
    int g_img_idx=flatten(col,row,w,h);
    int filter_sz=2*RAD+1;
    float channels[]{0.0,0.0,0.0};
    auto shared_mem_col=tid_x+RAD;
    auto shared_mem_row=tid_y+RAD;
    auto shared_mem_w=blockDim.x+2*RAD;
    auto shared_mem_h=blockDim.y+2*RAD;
    int shared_mem_idx=flatten(shared_mem_col,shared_mem_row,shared_mem_w,shared_mem_h);
    shared_mem[shared_mem_idx]=in[g_img_idx];
    if (tid_x<RAD && tid_y<RAD)
    {
        int shared_mem_halo_idx_up=flatten(shared_mem_col-RAD,shared_mem_row-RAD,shared_mem_w,shared_mem_h);
        int in_halo_idx_up=flatten(col-RAD,row-RAD,w,h);
        int shared_mem_halo_idx_down=flatten(shared_mem_col+blockDim.x,shared_mem_row-RAD,shared_mem_w,shared_mem_h);
        int in_halo_idx_down=flatten(col+blockDim.x,row-RAD,w,h);
        int shared_mem_halo_idx_left=flatten(shared_mem_col-RAD,shared_mem_row+blockDim.y,shared_mem_w,shared_mem_h);
        int in_halo_idx_left=flatten(col-RAD,row+blockDim.y,w,h);
        int shared_mem_halo_idx_right=flatten(shared_mem_col+blockDim.x,shared_mem_row+blockDim.y,shared_mem_w,shared_mem_h);
        int in_halo_idx_right=flatten(col+blockDim.x,row+blockDim.y,w,h);
        shared_mem[shared_mem_halo_idx_up]=in[in_halo_idx_up];
        shared_mem[shared_mem_halo_idx_down]=in[in_halo_idx_down];
        shared_mem[shared_mem_halo_idx_left]=in[in_halo_idx_left];
        shared_mem[shared_mem_halo_idx_right]=in[in_halo_idx_right];
    }
    if (tid_x<RAD)
    {
        int shared_mem_halo_idx_up=flatten(shared_mem_col-RAD,shared_mem_row,shared_mem_w,shared_mem_h);
        int in_halo_idx_up=flatten(col-RAD,row,w,h);
        int shared_mem_halo_idx_right=flatten(shared_mem_col+blockDim.x,shared_mem_row,shared_mem_w,shared_mem_h);
        int in_halo_idx_right=flatten(col+blockDim.x,row,w,h);
        shared_mem[shared_mem_halo_idx_up]=in[in_halo_idx_up];
        shared_mem[shared_mem_halo_idx_right]=in[in_halo_idx_right];
    }
    if (tid_y<RAD)
    {
        int shared_mem_halo_idx_down=flatten(shared_mem_col,shared_mem_row-RAD,shared_mem_w,shared_mem_h);
        int in_halo_idx_down=flatten(col,row-RAD,w,h);
        int shared_mem_halo_idx_left=flatten(shared_mem_col,shared_mem_row+blockDim.y,shared_mem_w,shared_mem_h);
        int in_halo_idx_left=flatten(col,row+blockDim.y,w,h);
        shared_mem[shared_mem_halo_idx_down]=in[in_halo_idx_down];
            shared_mem[shared_mem_halo_idx_left]=in[in_halo_idx_left];
    }
    __syncthreads();
    for (int r=-RAD;r<=RAD;r++)
    {
        for (int c=-RAD;c<=RAD;c++)
        {
            int shared_mem_local_idx=flatten(shared_mem_col+c,shared_mem_row+r,shared_mem_w,shared_mem_h);
            int filter_idx=flatten(RAD+c,RAD+r,filter_sz,filter_sz);
            channels[0]+=shared_mem[shared_mem_local_idx].x*filter[filter_idx];
            channels[1]+=shared_mem[shared_mem_local_idx].y*filter[filter_idx];
            channels[2]+=shared_mem[shared_mem_local_idx].z*filter[filter_idx];
        }

    }
    out[g_img_idx].x=clip(channels[0]);
    out[g_img_idx].y=clip(channels[1]);
    out[g_img_idx].z=clip(channels[2]);

}


int div_up(int a,int b)
{
    return (a+b-1)/b;

}

void sharpen_image(uchar4 *img_ptr,int w,int h,bool use_shared_mem_imp)
{
    size_t img_bytes=w*h*sizeof(uchar4);
    const int filter_sz=2*RAD+1;
    const auto filter_bytes= filter_sz*filter_sz*sizeof(float);
    const dim3 blocks(TX,TY);
    const dim3 grid(div_up(w,blocks.x),div_up(h,blocks.y));
    const float filter[9]={
        -1.0,-1.0,-1.0,
        -1.0,9.0,-1.0,
        -1.0,-1.0,-1.0,
    };
    uchar4 *d_in=nullptr,*d_out=nullptr;
    float *d_filter=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in),img_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),img_bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_filter),filter_bytes);
    cudaMemcpy(d_in,img_ptr,img_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter,filter,filter_bytes,cudaMemcpyHostToDevice);
    if (use_shared_mem_imp)
    {
        size_t shared_mem_sz=(TX+2*RAD)*(TY+2*RAD)*sizeof(uchar4);
        single_shared_mem_sharpen_kernel<<<grid,blocks,shared_mem_sz>>>(d_out,d_in,filter,w,h);
    }
    else
    {

        normal_sharpen_kernel<<<grid,blocks>>>(d_out,d_in,d_filter,w,h);

    }
    cudaDeviceSynchronize();
    cudaMemcpy(img_ptr,d_out,img_bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_filter),cudaFree(d_in),cudaFree(d_out);
}
