#include <bits/stdc++.h>
#include <cuda_runtime.h>
__global__ void kernel(int stream_idx)
{
    auto tgid=threadIdx.x+blockDim.x*blockIdx.x;
    if (tgid==0)
    {
        for (int i=0;i<25;i++)
        {
            printf("test stream with stream: %d\n",stream_idx);
        }
    }
}

int main()
{
    const auto sz=1<<25;
    const dim3 block(128);
    const dim3 grid(sz/block.x);
    std::array<cudaStream_t,3>streams;
    for (int i=0;i<streams.size();i++)
    {
        if (i==2)
        {
            cudaStreamCreate(&streams[i]);
        }
        else
        {
            cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
        }
    }

    kernel<<<grid,block,0,streams[0]>>>(1);
    cudaStreamSynchronize(streams[0]);
    kernel<<<grid,block,0,streams[1]>>>(2);
    cudaStreamSynchronize(streams[1]);
    kernel<<<grid,block,0,streams[2]>>>(3);
    cudaStreamSynchronize(streams[2]);
    cudaDeviceReset();
    return 0;
}