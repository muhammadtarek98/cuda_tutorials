#include "ray_casting.cuh"
__device__ uchar4 ray_caster(float *d_vol,int3 vol_size,Ray ray,float dist)
{
    uchar4 shade{make_uchar4(96,0,192,0)};
    float3 pos{ray.orign};
    float len = length(ray.dir);
    if (len < 1e-6f)
        return shade;
    float t=0.0f;
    float f=density(d_vol,vol_size,pos);
    while (f>dist+EPS &&t<1.0f)
    {
        f=density(d_vol,vol_size,pos);
        t+=(f-dist)/len;
        pos=param_ray(ray,t);
        f=density(d_vol,vol_size,pos);
    }
    if (t<1.0f)
    {
        const float3 ux{make_float3(1,0,0)},uy{make_float3(0,1,0)},uz{make_float3(0,0,1)};
        const auto grad_ux{(density(d_vol,vol_size,pos+EPS*ux)-density(d_vol,vol_size,pos))/EPS};
        const auto grad_uy{(density(d_vol,vol_size,pos+EPS*uy)-density(d_vol,vol_size,pos))/EPS};
        const auto grad_uz{(density(d_vol,vol_size,pos+EPS*uz)-density(d_vol,vol_size,pos))/EPS};
        const float3 grad{make_float3(grad_ux,grad_uy,grad_uz)};
        auto dir_normalized{normalize(ray.dir)};
        auto grad_normalized{normalize(grad)};
        float intensity=(-1*dot(dir_normalized,grad_normalized));
        shade=make_uchar4(255*intensity,0,0,255);
    }
    return shade;
}