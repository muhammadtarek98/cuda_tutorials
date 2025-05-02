#include <bits/stdc++.h>
#define  H2D cudaMemcpyHosttoDevice
#define  D2H cudaMemcpyDevicetoHost

__global__ void array_addition(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
void random_ints(int* arr, int size) {
    // Seed the random number generator
    srand(time(NULL));

    // Fill the array with random integers
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % 100;  // Generate random numbers between 0 and 99
    }
}
int main()
{

    int *a,*b,*c;
    int *d_a,*d_b,*d_c;
    int N = 512;
    int size=N*sizeof(int);
    a=(int*)malloc(size);
    b=(int*)malloc(size);
    c=(int*)malloc(size);
    random_ints(a,N);
    random_ints(b,N);
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    array_addition<<<N,1>>>(d_a,d_b,d_c);
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++)
    {
        std::cout<<c[i]<<" ";
    }
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}