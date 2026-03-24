#include "operations.cuh"

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

__host__ __device__ float length(float3 vector)
{
    return sqrtf(dot(vector,vector));
}
__host__ __device__ float length(float4 vector)
{
    return sqrtf(dot(vector,vector));

}