#include <bits/stdc++.h>
void random_ints(int* arr, int size) {
    srand(time(NULL));
    for(int i = 0; i < size; i++) {
        arr[i] = int(rand() % 100);
    }
}

__global__ void add(int *a,int *b,int *c)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    c[idx]=a[idx]+b[idx];
}
void print(int* arr,int size)
{
    for (int i=0;i<size;i++)
    {
        std::cout<<arr[i]<<" ";
    }
}
int main()
{
    int N=2048*2048;
    int n_thread_per_block=512;
    int size=N*sizeof(int);
    int *a=nullptr, *b=nullptr, *c=nullptr;
    int *d_a=nullptr,*d_b=nullptr,*d_c=nullptr;
    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void **)&d_c,size);
    a=(int *)malloc(size);random_ints(a,N);
    b=(int *)malloc(size);random_ints(b,N);
    c=(int *)malloc(size);
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    add<<<N/n_thread_per_block,n_thread_per_block>>>(d_a,d_b,d_c);
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    print(c,N);
    free(a),free(b),free(c);
    cudaFree(d_a),cudaFree(d_b),cudaFree(d_c);

    return 0;
}