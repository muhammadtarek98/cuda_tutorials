#include <bits/stdc++.h>
#include <cuda_runtime.h>
#define gpuErrrchk(res){gpu_raise((res),__FILE__,__LINE__);}
inline void gpu_raise(cudaError_t code,const char *file,int line,bool terminate=true)
{
    if (code!=cudaSuccess)
    {
        std::cerr<<stderr<<"GPU error:\n"<<cudaGetErrorString(code)<<file <<line<<"\n";
        if (terminate)
        {
            exit(code);
        }
    }
}
extern "C"{
    __global__ void kernel(int *a,int *b,int *c,int sz)
    {
        int gid=blockIdx.x*blockDim.x+threadIdx.x;
        if (gid<sz){
            c[gid]+=a[gid]+b[gid];
            printf("value:%d, threadidx:%d, blockidx:%d \n",c[gid],threadIdx.x,blockIdx.x);
        }
    }
}
void random_ints(int* arr, int size) {
    srand(time(NULL));

    for(int i = 0; i < size; i++) {
        arr[i] = rand();
    }
}
int main()
{
    int sz=128,n_blocks=2;
    size_t nbytes=sz*sizeof(int);
    clock_t gpu_start,gpu_end;
    cudaError cuda_error;
    dim3 blocks(sz/n_blocks);
    dim3 grids(n_blocks);
    int *h_a,*h_b,*h_c,*d_a,*d_b,*d_c;
    h_a=(int*)malloc(nbytes);
    h_b=(int*)malloc(nbytes);
    h_c=(int*)malloc(nbytes);
    memset(h_c,0,nbytes);
    random_ints(h_a,sz);
    random_ints(h_b,sz);
    gpuErrrchk(cudaMalloc((void**)&d_a,nbytes));
    gpuErrrchk(cudaMalloc((void**)&d_b,nbytes));
    gpuErrrchk(cudaMalloc((void**)&d_c,nbytes));
    gpuErrrchk(cudaMemcpy(d_a,h_a,nbytes,cudaMemcpyHostToDevice));
    gpuErrrchk(cudaMemcpy(d_b,h_b,nbytes,cudaMemcpyHostToDevice));
    gpu_start=clock();
    kernel<<<grids,blocks>>>(d_a,d_b,d_c,sz);
    cudaDeviceSynchronize();
    gpuErrrchk(cudaMemcpy(h_c,d_c,nbytes,cudaMemcpyDeviceToHost));
    gpu_end=clock();
    std::cout<<(double)(gpu_end-gpu_start)/CLOCKS_PER_SEC<<" seconds"<<std::endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
    delete h_a;
    delete h_b;
    delete h_c;
    return 0;
}