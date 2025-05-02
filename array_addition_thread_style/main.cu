#include <bits/stdc++.h>
void random_ints(int* arr, int size) {
    srand(time(NULL));

    for(int i = 0; i < size; i++) {
        arr[i] = rand();
    }
}
__global__ void add(int *a,int *b,int *c)
{
    c[threadIdx.x]=a[threadIdx.x]+b[threadIdx.x];
}
int main()
{
    int N=512;
    int size=sizeof(int)*N;
    int *a=(int*)malloc(size);
    int*b=(int*)malloc(size);
    int*c=(int*)malloc(size);
    int*d_a,*d_b,*d_c;
    random_ints(a,N);
    random_ints(b,N);
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    add<<<1,N>>>(d_a,d_b,d_c);
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    cudaFree(d_a),cudaFree(d_b),cudaFree(d_c);
    for (int i=0;i<size;++i)
    {
        std::cout<<c[i]<<"\n";
    }
    free(a);free(b);free(c);

    return 0;
}