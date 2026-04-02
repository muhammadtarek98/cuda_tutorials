#include "operations.cuh"
__host__ __device__ int div_up(int a,int b)
{
    return (a+b-1)/b;
}
__host__ __device__ float dot(float3 x,float3 y)
{
    return x.x*y.x+y.y*x.y+x.z*y.z;
}
__host__ __device__ float dot(float4 x,float4 y)
{
    return x.x*y.x+y.y*x.y+x.z*y.z+x.w*y.w;
}
__host__ __device__ float dot(int3 x,int3 y)
{
    return x.x*y.x+y.y*x.y+x.z*y.z;
}
__host__ __device__ float dot(uint3 x,uint3 y)
{
    return x.x*y.x+y.y*x.y+x.z*y.z;
}
__host__ __device__ float3 normalize(float3 x)
{
    auto invlen=rsqrtf(dot(x,x));
    return x*invlen;
}

__host__ __device__ float fracf(float v)
{
    return v - std::floor(v);
}


__host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
__host__ __device__ float length(float3 vector)
{
    return sqrtf(dot(vector,vector));
}
__host__ __device__ float length(float4 vector)
{
    return sqrtf(dot(vector,vector));

}

__host__ __device__ float3 srcIdxToPos(int c,int r,int w,int h,float zs)
{
return make_float3(c-(w/2),r-(h/2),zs);
}
__host__ __device__ int3 posToVolIdx(float3 pos,int3 vol_size){
return make_int3(pos.x+(vol_size.x/2),pos.y+(vol_size.y/2),pos.z+(vol_size.z/2));
}
__host__ __device__ int flatten(int3 idx,int3 vol_size){
return idx.x+idx.y*vol_size.x+idx.z*vol_size.x*vol_size.y;
}
__host__ __device__ float3 yRotate(float3 pos,float theta)
{
    float c=cosf(theta),s=sinf(theta);
    return make_float3(pos.x*c+pos.z*s,pos.y,-s*pos.x+c*pos.z);

}