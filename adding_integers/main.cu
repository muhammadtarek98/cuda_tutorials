#include <iostream>
__global__ void add(const int *a,const int *b,int *c)
{
    *c=*a+*b;
}
int main()
{
    //create host data
    int a=7,b=8,c=0;
    //create device pointers for host data
    int *d_a=&a,*d_b=&b,*d_c=&c;
    //get the size of the host data
    int size=sizeof(int);
    cudaMalloc((void**)&d_a,size);
    //allocate device memory
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);
    //copy the data from host to device
    cudaMemcpy(d_a,&a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,size,cudaMemcpyHostToDevice);
    //execute the kernel function
    add<<<1,1>>>(d_a,d_b,d_c);
    //copy the data from device to host
    cudaMemcpy(&c,d_c,size,cudaMemcpyDeviceToHost);
    std::cout<<c<<std::endl;
    //deallocate the device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}