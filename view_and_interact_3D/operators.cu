#include "operators.cuh"
__host__ __device__ void operator+=(float3 &a, float3 &b)
{
    a.x+=b.x;
    a.y+=b.y;
    a.z+=b.z;
}
__host__ __device__ void operator+=(float3 &a, float b)
{
    a.x+=b;
    a.y+=b;
    a.z+=b;
}

__host__ __device__ void operator+=(int3 &a, int3 &b)
{
    a.x+=b.x;
    a.y+=b.y;
    a.z+=b.z;
}
__host__ __device__ void operator+=(int3 &a, int b)
{
    a.x+=b;
    a.y+=b;
    a.z+=b;
}
__host__ __device__ void operator-=(float3 &a, float3 &b)
{
    a.x-=b.x;
    a.y-=b.y;
    a.z-=b.z;
}
__host__ __device__ void operator-=(float3 &a, float b)
{
    a.x-=b;
    a.y-=b;
    a.z-=b;
}
__host__ __device__ void operator-=(int3 &a, int3 &b)
{
    a.x-=b.x;
    a.y-=b.y;
    a.z-=b.z;
}
__host__ __device__ void operator-=(int3 &a, int b)
{
    a.x-=b;
    a.y-=b;
    a.z-=b;
}
__host__ __device__ float3 operator+(const float3 a, const float b)
{
    return make_float3(a.x+b,a.y+b,a.z+b);
}
__host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x+b,a.y+b,a.z+b);
}
__host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x+b,a.y+b,a.z+b);
}
__host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x+b.x,a.y+b.y,a.z+b.z);
}
__host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x+b,a.y+b,a.z+b);

}
__host__ __device__ float3 operator+(const float a,const float3 &b)
{
    return make_float3(a+b.x,a+b.y,a+b.z);
}
__host__ __device__ float3 operator-( float a,const float3 &b)
{
    return make_float3(a-b.x,a-b.y,a-b.z);
}
__host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
}

__host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x-b,a.y-b,a.z-b);
}
__host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(a.x-b,a.y-b,a.z-b);
}

__host__ __device__ int3 operator-(int3 a, int3 b)
{
    return  make_int3(a.x-b.x,a.y-b.y,a.z-b.z);
}

__host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x-b,a.y-b,a.z-b);
}

__host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(a.x-b,a.y-b,a.z-b);
}
__host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x*b.x,a.y*b.y,a.z*b.z);
}
__host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x/b.x,a.y/b.y,a.z/b.z);
}
__host__ __device__ float operator*(float a,float3 b)
{
    return a*b.x+a*b.y+a*b.z;
}
__host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x*b,a.y*b,a.z*b);
}
__host__ __device__ uint3 operator*(uint3 a, float b)
{
    return make_uint3(a.x*b,a.y*b,a.z*b);
}