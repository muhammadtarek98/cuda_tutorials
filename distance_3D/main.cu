#include <bits/stdc++.h>
#define TX 8
#define TY 8
#define TZ 8
#define W 32
#define H 32
#define Z 32
__device__ float distance(int x,int y,int z,float3 pos)
{
    const auto x_dist_sqr=(x-pos.x)*(x-pos.x);
    const auto y_dist_sqr=(y-pos.y)*(y-pos.y);
    const auto z_dist_sqr=(z-pos.z)*(z-pos.z);

    return sqrtf(x_dist_sqr+y_dist_sqr+z_dist_sqr);

}
__global__ void kernel(float *out,int w,int h,int s,float3 pos)
{
    const auto tgidx=threadIdx.x+blockDim.x*blockIdx.x;
    const auto tgidy=threadIdx.y+blockDim.y*blockIdx.y;
    const auto tgidz=threadIdx.z+blockDim.z*blockIdx.z;
    const auto idx=tgidx+tgidy*w+s*w*h;
    if (tgidx>=w||tgidy>=h||tgidz>=s){return;}
    out[idx]=distance(tgidx,tgidy,tgidz,pos);
}
__host__ int div_up(int a,int b)
{
    return (a+b-1)/b;
}
int main()
{
    const dim3 block(TX,TY,TZ);
    const dim3 grid(div_up(W,TX),div_up(H,TY),div_up(Z,TZ));
    float3 pos{0.0,0.0,0.0};
    const auto bytes{H*W*Z*sizeof(float)};
    std::unique_ptr<float[]>h_out{std::make_unique<float[]>(W*H*Z)};
    float *d_out=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_out),bytes);
    kernel<<<grid,block>>>(d_out,W,H,Z,pos);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.get(),d_out,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return 0;
}