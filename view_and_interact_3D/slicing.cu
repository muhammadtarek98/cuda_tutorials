#include "slicing.cuh"
__device__ int3 posToIdx(const float3 pos, const int3 vol_size)
{
    return make_int3(pos.x+vol_size.x/2,pos.y+vol_size.y/2,pos.z+vol_size.z/2);
}
__device__ int clip(const int n, const int n_max, const int n_min)
{
    return n>n_max?n_max:(n<n_min?n_min:n);
}
__host__ __device__ float fracf(float v)
{
    return v - std::floor(v);
}
__device__ int flatten(int3 index, int3 volSize) {
    return index.x + index.y*volSize.x + index.z*volSize.x*volSize.y;
}

__host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
__device__ float density(float *d_vol,int3 vol_size,float3 pos)
{
    int3 idx=posToIdx(pos,vol_size);
    int i{idx.x},j{idx.y},k{idx.z};
    const int w{vol_size.x},h{vol_size.y},d{vol_size.z};
    idx=make_int3(clip(i,0,w-2),clip(j,0,h-2),clip(k,0,d-2));
    const float3 reminder=fracf(pos);
    int3 dx{make_int3(1,0,0)},dy{make_int3(0,1,0)},dz{make_int3(0,0,1)};
    const float dens000{d_vol[flatten(idx,vol_size)]};
    const float dens100{d_vol[flatten(idx+dx,vol_size)]};
    const float dens010{d_vol[flatten(idx+dy,vol_size)]};
    const float dens001{d_vol[flatten(idx+dz,vol_size)]};
    const float dens110{d_vol[flatten(idx+dx+dy,vol_size)]};
    const float dens101{d_vol[flatten(idx+dx+dz,vol_size)]};
    const float dens011{d_vol[flatten(idx+dy+dz,vol_size)]};
    const float dens111{d_vol[flatten(idx+dx+dy+dz,vol_size)]};
    float x1=(1-reminder.x)*(1-reminder.y)*(1-reminder.z)*dens000;
    float x2=reminder.x*(1-reminder.y)*(1-reminder.z)*dens100;
    float x3=(1-reminder.x)*reminder.y*(1-reminder.z)*dens010;
    float x4=(1-reminder.x)*(1-reminder.y)*reminder.z*dens001;
    float x5=reminder.x*reminder.y*(1-reminder.z)*dens110;
    float x6=reminder.x*(1-reminder.y)*reminder.z*dens101;
    float x7=reminder.x*reminder.y*reminder.z*dens111;
    float x8=(1-reminder.x)*reminder.y*reminder.z*dens011;
    return x1+x2+x3+x4+x5+x6+x7+x8;
}
__device__ float planeSDF(float3 pos,float3 norm,float d)
{
    return dot(pos,normalize(norm))-d;
}
__device__ bool ray_plane_intersect(Ray ray,float3 n,float dist,float *t)
{
    const float f0{planeSDF(param_ray(ray,0.0f),n,dist)};
    const float f1{planeSDF(param_ray(ray,1.0f),n,dist)};
    bool flag{0*f1<0};
    if (flag)
    {
        *t=(0-f0)/(f1-f0);
    }
    return flag;
}
__device__ uchar4 slice_shader(float *d_vol,int3 vol_size,Ray ray,float gain,float dist,float3 norm)
{
    uchar4 shade=make_uchar4(96,0,192,0);
    float t;
    if (ray_plane_intersect(ray,norm,dist,&t))
    {
        float3 ray_pos=param_ray(ray,t);
        float slice_dens=density(d_vol,vol_size,ray_pos);
        shade=make_uchar4(48,clip(-10.0f*(1.0+gain)*slice_dens,0,255),96,255);
    }
    return shade;
}