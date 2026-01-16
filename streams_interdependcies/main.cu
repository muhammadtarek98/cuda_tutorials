#include <cuda_runtime.h>
#include <bits/stdc++.h>
const auto sz=1<<15;
const dim3 block(128);
const dim3 grid(sz/block.x);
std::array<cudaStream_t,3> streams;
cudaEvent_t start;

__global__ void kernel(int stream_idx)
{
    int tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid==0)
    {
        printf("test events with stream:%d\n",stream_idx);
    }
}
int main()
{

    for (auto &stream:streams)
    {
        cudaStreamCreate(&stream);
    }
    cudaEventCreateWithFlags(&start,cudaEventDisableTiming);
    kernel<<<grid,block,0,streams[0]>>>(0);
    cudaEventRecord(start,streams[0]);
    cudaStreamWaitEvent(streams[2],start,0);
    kernel<<<grid,block,0,streams[1]>>>(1);
    kernel<<<grid,block,0,streams[2]>>>(2);
    cudaEventDestroy(start);
    for (auto&stream:streams)
    {
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}