#include "ray_structure.cuh"
__device__ float3 param_ray(Ray r,float t)
{
    auto origin=r.orign;
    auto dir=r.dir;
    return origin+(t*dir);
}
__device__ bool intersection(Ray r,float3 box_min,float3 box_max,float *t_near,float *t_far)
{
    const float3 invR{make_float3(1.0f,1.0f,1.0f)/r.dir};
    const float3 tbottom{invR*(box_min-r.orign)};
    const float3 ttop{invR*(box_max-r.orign)};
    const float3 tmin{make_float3(min(ttop.x,tbottom.x),min(ttop.y,tbottom.y),min(ttop.z,tbottom.z))};
    const float3 tmax{make_float3(max(ttop.x,tbottom.x),max(ttop.y,tbottom.y),max(ttop.z,tbottom.z))};
    *t_near = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    *t_far  = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
    return *t_far>*t_near;
}