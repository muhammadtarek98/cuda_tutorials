#include "kernel.cuh"

 int div_up(int a, int b)
{
    return (a+b-1)/b;
}

__device__ unsigned char clip(int n)
{
    return n>255?255:(n<0?0:n);
}
__device__ int idxclip(int idx,int idx_max)
{
    return idx >(idx_max-1)?(idx_max-1):(idx<0?0:idx);

}
__device__ int flatten(int col,int row,int width,int height){
    return idxclip(col,width)+idxclip(row,height)*width;

}
__device__ float dist_sqr(int x,int y,int row,int col)
{
    return ((col-x)*(col-x))+((row-y)*(row-y));
}
__global__ void temp_kernel(uchar4 *d_out,float *d_temp, int w,int h,BC bc)
{
    extern __shared__ float s_in[];
    int col=threadIdx.x+blockDim.x*blockIdx.x;
    int row=threadIdx.y+blockDim.y*blockIdx.y;
    if ((col>=w)||(row>=h))return;
    const int idx=flatten(col,row,w,h);
    const  int sm_w=blockDim.x+2*RAD;
    const int sm_h=blockDim.y+2*RAD;
    const int sm_col=threadIdx.x+RAD;
    const int sm_row=threadIdx.y+RAD;
    const int sm_idx=flatten(sm_col,sm_row,sm_w,sm_h);
    d_out[idx].x=0;
    d_out[idx].y=0;
    d_out[idx].z=0;
    d_out[idx].w=255;
    s_in[sm_idx]=d_temp[idx];
    if (threadIdx.x<RAD)
    {
        s_in[flatten(sm_col-RAD,sm_row,sm_w,sm_h)]=
            d_temp[flatten(col-RAD,row,w,h)];
        s_in[flatten(sm_col+blockDim.x,sm_row,sm_w,sm_h)]=
            d_temp[flatten(col+blockDim.x,row,w,h)];
    }
    if (threadIdx.y<RAD)
    {
        s_in[flatten(sm_col,sm_row-RAD,sm_w,sm_h)]=
            d_temp[flatten(col,row-RAD,w,h)];
        s_in[flatten(sm_col,sm_row+blockDim.y,sm_w,sm_h)]=
            d_temp[flatten(col,row+blockDim.y,w,h)];
    }
    float dist=((col-bc.X)*(col-bc.X))+((row-bc.Y)*(row-bc.Y));
    if (dist<bc.rad*bc.rad)
    {
        d_temp[idx]=bc.t_s;
        d_out[idx].x = 255;
        d_out[idx].y = 0;
        d_out[idx].z = 0;
        return;
    }
    bool plate_flag=(col==0)||(col==w-1)||(row==0)||(col+row<bc.Chamfer)||(col-row>w-bc.Chamfer);
    if (plate_flag)
    {
        d_temp[idx]=bc.t_a;
        return;
    }
    if (row==h-1)
    {
        d_temp[idx]=bc.t_g;
        return;
    }
    __syncthreads();
    int pre_final_col_idx=flatten(sm_col-1,sm_row,sm_w,sm_h);
    int final_col_idx=flatten(sm_col+1,sm_row,sm_w,sm_h);
    int pre_final_row_idx=flatten(sm_col,sm_row-1,sm_w,sm_h);
    int final_row_idx=flatten(sm_col,sm_row+1,sm_w,sm_h);

    float temp=0.25f*(s_in[pre_final_col_idx]+s_in[final_col_idx]+s_in[pre_final_row_idx]+s_in[final_row_idx]);
    d_temp[idx]=temp;
    const unsigned char intensity=clip(static_cast<int>(temp));
    d_out[idx].x=intensity;
    d_out[idx].z=255-intensity;

}

__global__ void reset_kernel(float *d_temp,int w,int h,BC bc)
{
  const int col = blockIdx.x*blockDim.x + threadIdx.x;
  const int row = blockIdx.y*blockDim.y + threadIdx.y;
  if ((col >= w) || (row >= h)) return;
  d_temp[row*w + col] = bc.t_a;

}
 void reset_temp(float *d_temp,int w,int h,BC bc)
{
    const dim3 block(TY,TX);
    const dim3 grid(div_up(w,TX),div_up(h,TY));
    reset_kernel<<<grid,block>>>(d_temp,w,h,bc);
    cudaDeviceSynchronize();


}
 void kernel_launcher(uchar4 *d_out,float *d_temp, int w,int h,BC bc)
{
    const dim3 block(TX,TY);
    const dim3 grid(div_up(w,block.x),div_up(h,block.y));
    const auto shared_mem_sz=(block.x+2*RAD)*(block.y+2*RAD)*sizeof(float);
    temp_kernel<<<grid,block,shared_mem_sz>>>(d_out,d_temp,w,h,bc);
    cudaDeviceSynchronize();


}
