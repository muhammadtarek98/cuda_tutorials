#include <bits/stdc++.h>
#include <cuda_runtime.h>
const int block_size=2;
struct Configs
{
    dim3 dimsA,dimsB,dimsC;
    int mem_size_A,mem_size_B,mem_size_C;
    int size_a,size_b,size_c;
    int h2d,d2h;
    dim3 threads,grid;
    Configs()
    {
        this->dimsA=dim3(5 * 2 * block_size, 5 * 2 * block_size, 1);
        this->dimsB=dim3(5 * 4 * block_size, 5 * 2 * block_size, 1);

        this->dimsC=dim3(this->dimsB.x, this->dimsA.y, 1);
        this->size_a=static_cast<int>(this->dimsA.x * this->dimsA.y);
        this->size_b=static_cast<int>(this->dimsB.x * this->dimsB.y);
        this->size_c=static_cast<int>(this->dimsC.x * this->dimsC.y);
        this->mem_size_A=static_cast<int>(this->size_a*sizeof(int));
        this->mem_size_B=static_cast<int>(this->size_b*sizeof(int));
        this->mem_size_C=static_cast<int>(this->size_c*sizeof(int));
        this->h2d=cudaMemcpyHostToDevice;
        this->d2h=cudaMemcpyDeviceToHost;
        this->threads=dim3(block_size, block_size);
        this->grid=dim3(this->dimsB.x / this->threads.x, this->dimsA.y / this->threads.y);
    }
};

__global__ void MatMul(int *A,int*B,int*C, int xA, int xB)
{
    auto bx=blockIdx.x;
    auto by=blockIdx.y;
    auto tx=threadIdx.x;
    auto ty=threadIdx.y;
    auto a_begin=xA*block_size*by;
    auto a_end=a_begin+xA-1;
    auto b_begin=xB*block_size;
    auto a_step=block_size,b_step=block_size*xB;
    int C_sub=0;

    for (int a=static_cast<int>(a_begin),b=(b_begin);a<=a_end;a+=a_step,b+=b_step)
    {
        __shared__ int As[block_size][block_size];
        __shared__ int Bs[block_size][block_size];
        As[ty][tx]=A[a+xA*ty+tx];
        Bs[ty][tx]=B[b+xB*ty+tx];
        __syncthreads();
        for (int k=0;k<block_size;++k)
        {
            C_sub+=As[ty][k]*Bs[k][tx];
        }
        __syncthreads();

    }

    auto c_step=xB*block_size*by+block_size*bx;
    C[c_step+xB*ty+tx]=C_sub;
}
void print(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}
void random_ints(int* arr, int size) {
    srand(time(NULL));
    for(int i = 0; i < size; i++) {
        arr[i] = int(rand() % 10);
    }
}
int main()
{
    int devID=0;
    cudaSetDevice(devID);
    Configs config;
std::cout << "dimsA: " << config.dimsA.x << " x " << config.dimsA.y << " x " << config.dimsA.z << "\n";
std::cout << "dimsB: " << config.dimsB.x << " x " << config.dimsB.y << " x " << config.dimsB.z << "\n";
    std::cout << "dimsC: " << config.dimsC.x << " x " << config.dimsC.y << " x " << config.dimsC.z << "\n";

    int*matA =static_cast<int*>(malloc(config.mem_size_A));
    int*matB =static_cast<int*>(malloc(config.mem_size_B));
    int*matC=static_cast<int*>(malloc(config.mem_size_C));
    random_ints(matA, config.size_a);
    random_ints(matB, config.size_b);
    print(matA,config.size_a);
    print(matB,config.size_b);
    int *d_matA=nullptr,*d_matB=nullptr,*d_matC=nullptr;
    cudaMalloc((void**)&d_matA,config.mem_size_A);
    cudaMalloc((void**)&d_matB,config.mem_size_B);
    cudaMalloc((void**)&d_matC,config.mem_size_C);
    cudaMemcpy(d_matA,matA,config.mem_size_A,static_cast<cudaMemcpyKind>(config.h2d));
    cudaMemcpy(d_matB,matB,config.mem_size_B,static_cast<cudaMemcpyKind>(config.h2d));
    MatMul<<<config.grid,config.threads>>>(d_matA,d_matB,d_matC,static_cast<int>(config.dimsA.x),static_cast<int>(config.dimsB.x));
    cudaMemcpy(matC,d_matC,config.mem_size_C,static_cast<cudaMemcpyKind>(config.d2h));
    print(matC,config.size_c);
    free(matA),free(matB),free(matC);
    cudaFree(d_matA),cudaFree(d_matB),cudaFree(d_matC);

    return 0;

}