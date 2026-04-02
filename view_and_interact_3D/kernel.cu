#include "kernel.cuh"

#include "ray_casting.cuh"
#include "volume_render.cuh"
__device__ float compute_and_store(int c,int r,int s,int id,int3 vol_size,float4 params)
{
    int3 pos0{vol_size.x/2,vol_size.y/2,vol_size.z/2};
    float dx=c-pos0.x,dy=r-pos0.y,dz=s-pos0.z;
    if (id==0)
    {
        return sqrtf(dx*dx+dy*dy+dz*dz)-params.x;
    }
    else if (id==1)
    {
        const float r=sqrtf(dx*dx+dy*dy);
        return sqrtf((r-params.x)*(r-params.x)+dz*dz)-params.y;
    }
    else
    {
        float x=fabs(dx)-params.x,y=fabs(dy)-params.y,z=fabs(dz)-params.z;
        if (x<=0&&y<=0&&z<=0)
        {
            return fmaxf(x,fmaxf(y,z));
        }
        else
        {
            x=fmaxf(x,0.0f),y=fmaxf(y,0.0f),z=fmax(z,0.0f);
            return sqrtf(x*x+y*y+z*z);
        }
    }
}
__global__ void kernel_render(uchar4 *d_out,float *d_vol,int w,int h,int3 vol_size,int method,float zs,float theta,float threshold,float dist)
{
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    int r=blockDim.y*blockIdx.y+threadIdx.y;
    int i=c+r*w;
    if ((c>=w)||(r>=h))
    {
        return;
    }
    const uchar4 background{64,0,128,0};
    float3 source = { 0.f, 0.f, -zs };
    float3 pix=srcIdxToPos(c,r,w,h,2*vol_size.z-zs);
    source=yRotate(source,theta);
    pix=yRotate(pix,theta);
    float t0,t1;
    Ray pix_ray{source,pix-source};
    float3 center{vol_size.x/2.0f,vol_size.y/2.0f,vol_size.z/2.0f};
    float3 boxmin=center*-1;
    float3 boxmax{vol_size.x-center.x,vol_size.y-center.y,vol_size.z-center.z};
    bool box_intersection=intersection(pix_ray,boxmin,boxmax,&t0,&t1);
    uchar4 shade;
    if (!box_intersection)
    {
        shade=background;
    }
    else
    {
        if (t0<0.0f){t0=0.0f;}
        Ray box_ray{param_ray(pix_ray,t0),param_ray(pix_ray,t1)-param_ray(pix_ray,t0)};
        if (method==1)
        {
            shade=slice_shader(d_vol,vol_size,box_ray,threshold,dist,source);
        }
        else if (method==2)
        {
            shade=ray_caster(d_vol,vol_size,box_ray,threshold);
        }
        else
        {
            shade=volume_render_shader(d_vol,vol_size,box_ray,threshold,NUM);
        }
    }
    d_out[i]=shade;

}
__global__ void volume_kernel(float *d_vol,int3 vol_size,int id,float4 params)
{
    int w=vol_size.x,h=vol_size.y,d=vol_size.z;
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    int r=blockDim.y*blockIdx.y+threadIdx.y;
    int s=blockDim.z*blockIdx.z+threadIdx.z;
    int i=c+r*w+s*w*h;
    if ((c>=w)||(r>=h)||(s>=d))
    {
        return;
    }
    d_vol[i]=compute_and_store(c,r,s,id,vol_size,params);
}
void kernel_launch(uchar4 *d_out,float *d_vol,int w,int h,int3 vol_size,int method,int zs,float theta,float threshold,float dist)
{
    dim3 block(TX_2D,TY_2D);
    dim3 grid(div_up(w,TX_2D),div_up(h,TY_2D));
    kernel_render<<<grid,block>>>(d_out,d_vol,w,h,vol_size,method,zs,theta,threshold,dist);
}
void volume_kernel_launch(float *d_vol,int3 vol_size,int id,float4 params)
{
    dim3 block(TX,TY,TZ);
    dim3 grid(div_up(vol_size.x,TX),div_up(vol_size.y,TY),div_up(vol_size.z,TZ));
    volume_kernel<<<grid,block>>>(d_vol,vol_size,id,params);

}