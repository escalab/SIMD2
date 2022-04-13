#include <cuda_fp16.h>
/**
  Simple kernel wappers to convert precisioin
*/
__global__ void cuda_float2half(float *floatdata, __half *halfdata, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
		halfdata[index] = __float2half(floatdata[index]);
	}
}

__global__ void cuda_half2float(__half *halfdata, float *floatdata, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
      floatdata[index] = __half2float(halfdata[index]);
	}
}

void f2h_device(half * fp16_d, float * fp32_d, int num_elements){
    int blockSize = 1024;
    int numBlocks = ( num_elements + blockSize - 1) / blockSize;
    cuda_float2half<<<numBlocks, blockSize>>>(fp32_d, fp16_d, num_elements);
    cudaDeviceSynchronize();
};

void h2f_device(float * fp32_d, half * fp16_d, int num_elements){
  int blockSize = 1024;
  int numBlocks = ( num_elements + blockSize - 1) / blockSize;
  cuda_half2float<<<numBlocks, blockSize>>>(fp16_d, fp32_d, num_elements);
  cudaDeviceSynchronize();
};