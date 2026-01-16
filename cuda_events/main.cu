#include <iostream>
__global__ void kernel()
{
    double sum=0.0;
    for (int i=0;i<1000;i++)
    {
        sum+=tan(0.1)*tan(0.1);
    }
}
int main()
{
    const auto sz=1<<12;
    const dim3 blocks(128);
    const dim3 grid(sz/blocks.x);
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    kernel<<<grid,blocks,0>>>();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time,start,end);
    std::cout<<time<<" ms";
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaDeviceReset();
    return 0;
}