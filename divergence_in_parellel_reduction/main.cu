#include <bits/stdc++.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
enum INIT_PARAM{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};


void print_array(int * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%d,", input[i]);
		}
		else
		{
			printf("%d \n", input[i]);
		}
	}
}
//simple initialization
void init(std::vector<int>&arr)
{
	for (int i=0;i<(int)arr.size();++i)
	{
		arr[i]=i%10;
	}
}
void seq_array_accum(int &a, const std::vector<int> &arr)
{
	for (const auto &i : arr)
	{
		a+=i;
	}
}

bool compare_results(const int &cpu_res,const int &gpu_res)
{
	if (gpu_res!=cpu_res)
	{
		return false;
	}
	std::cout<<cpu_res<<"="<<gpu_res<<std::endl;
	return true;
}
void compare_arrays(float * a, float * b, float size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");

			return;
		}
	}
	printf("Arrays are same \n");

}
__global__ void interleaved_pair(int *input,int *part,const int size)
{
	const auto tid=threadIdx.x;
	const auto tgid=tid+blockDim.x*blockIdx.x;
	if (tgid>size){return;}
	for (auto offset=blockDim.x/2;offset>0;offset/=2)
	{
		if (tid<offset)
		{
			input[tgid]+=input[tgid+offset];
		}
		__syncthreads();
	}
	if (tid==0)
	{
		part[blockIdx.x]=input[tgid];
	}

}

__global__ void reduction_neighbored_pairs_improved(int *input,int *part,const int size)
{
	const auto tid = threadIdx.x;
	const auto tgid = blockDim.x * blockIdx.x + tid;
	//local window data
	int *window = input + blockDim.x * blockIdx.x;
	if (tgid > size)
		return;
	for (int offset = 1; offset <= blockDim.x /2 ; offset *= 2)
	{
		auto index = 2 * offset * tid;
		if (index < blockDim.x)
		{
			window[index] += window[index + offset];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		part[blockIdx.x] = input[tgid];
	}
}

int main()
{
	int size = 1 << 27,block_size = 128,cpu_res=0,gpu_res=0;
	dim3 block(block_size);
	dim3 grid(size / block.x);
	const auto input_byte_size = size * sizeof(int);
	const auto part_byte_size = grid.x*sizeof(int);
	std::vector<int> h_input(size);
	std::vector<int> h_part(grid.x);
 	init(h_input);
	int *d_input=nullptr;
	int *d_part=nullptr;
	cudaMalloc(reinterpret_cast<void**>(&d_input),input_byte_size);
	cudaMalloc(reinterpret_cast<void**>(&d_part),part_byte_size);
	cudaMemset(d_part,0,part_byte_size);
	seq_array_accum(cpu_res,h_input);
	cudaMemcpy(d_input,h_input.data(),input_byte_size,cudaMemcpyHostToDevice);
	interleaved_pair<<<grid,block>>>(d_input,d_part,size);
	cudaDeviceSynchronize();
	cudaMemcpy(h_part.data(),d_part,part_byte_size,cudaMemcpyDeviceToHost);
	seq_array_accum(gpu_res,h_part);
	std::cout<<compare_results(cpu_res,gpu_res)<<std::endl;
	cudaFree(d_input),cudaFree(d_part);


	return 0;
}