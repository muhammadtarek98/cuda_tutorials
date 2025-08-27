#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void sum_arrays_1Dgrid_1Dblock(float *a,float *b,float *c,int nx)
{
    auto tid=threadIdx.x+blockIdx.x;
    c[tid]=a[tid]+b[tid];
}
__global__ void sum_arrays_2Dgrid_2Dblock(float *a,float *b,float *c,int nx,int ny)
{
    auto tidx=blockIdx.x*blockDim.x+threadIdx.x;
    auto tidy=blockDim.y*blockIdx.y+threadIdx.y;
    auto tid=tidy*nx+tidx;
    if (tid<nx&&tid<ny)
        c[tid]=a[tid]+b[tid];

}
void sum_array_cpu(const float *a,const float *b,float *c,const int &size){
for (size_t i=0;i<size;++i)
{
    c[i]=a[i]+b[i];
}
}
void run_sum_array_1d(int argc,char const *argv[])
{
    int size=1<<22;
    int block_size=128;
    if (argc>2)
    {
        size=1<<atoi(argv[2]);
    }
    if (argc>4)
    {
        block_size=1<<atoi(argv[4]);
    }
    auto byte_size=size*sizeof(float);
    float *h_a,*h_b,*h_c;
    float *d_a,*d_b,*d_c;
    h_a=(float*)malloc(byte_size);
    h_b=(float*)malloc(byte_size);
    h_c=(float*)malloc(byte_size);
    if (!h_a)
    {
        std::cout<<"host allocation error\n";
    }
    for (size_t i =0;i<size;++i)
    {
        h_a[i]=i%10;
        h_b[i]=i%7;
    }
    sum_array_cpu(h_a,h_b,h_c,size);
    dim3 block(block_size);
    dim3 grid((size+block.x-1)/block.x);
    cudaMalloc(reinterpret_cast<void**>(&d_a),byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_b),byte_size);
    cudaMalloc(reinterpret_cast<void**>(&d_c),byte_size);
    cudaMemset(d_c,0,byte_size);
    cudaMemcpy(d_a,h_a,byte_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,byte_size,cudaMemcpyHostToDevice);
    sum_arrays_1Dgrid_1Dblock<<<grid,block_size>>>(h_a,h_b,h_c,size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,byte_size,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b),cudaFree(d_c);
    free(h_a),free(h_b),free(h_c);

}
void run_sum_array_2d(int argc,char const *argv[])
{
    int size=1<<22;
    int block_x=128;
int block_y=8;
    int nx=1<<14;
    int ny=size/nx;
    if (argc>4)
    {
        int pow=atoi(argv[4]);
        if (pow<3||pow>10)
        {
            std::cout<<"invalid configs \n";
        }
        else
        {
            block_x=1<<pow;
            block_y=1024/block_x;
        }
    }
    auto bytsize=size*sizeof(float);
    float *h_a,*h_b,*h_c;

    h_a=static_cast<float*>(malloc(bytsize));
    h_b=static_cast<float*>(malloc(bytsize));
    h_c=static_cast<float*>(malloc(bytsize));
    memset(h_c,0,bytsize);
    for (size_t i =0;i<size;++i)
    {
        h_a[i]=i%10;
        h_b[i]=i%7;
    }
    dim3 block_size(block_x,block_y);
    dim3 grid_size((nx+block_x-1)/block_x,(ny+block_y-1)/block_y);
    sum_array_cpu(h_a,h_b,h_c,size);
    float *d_a,*d_b,*d_c;
    cudaMalloc(reinterpret_cast<void**>(&d_a),bytsize);
    cudaMalloc(reinterpret_cast<void**>(&d_b),bytsize);
    cudaMalloc(reinterpret_cast<void**>(&d_c),bytsize);
    cudaMemset(d_c,0,bytsize);
    cudaMemcpy(d_a,h_a,bytsize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytsize,cudaMemcpyHostToDevice);
    sum_arrays_2Dgrid_2Dblock<<<grid_size,block_size>>>(d_a,d_b,d_c,nx,ny);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,bytsize,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b),cudaFree(d_c);
    free(h_a),free(h_b),free(h_c);

}
int main(const int argc,char const *argv[])
{
    if (argc>1)
    {
        if (atoi(argv[1])>0)
        {
            run_sum_array_2d(argc,argv);
        }
        else
        {
            run_sum_array_1d(argc,argv);
        }
    }
    else
    {
        run_sum_array_1d(argc,argv);
    }


    return 0;
}