#include <stdio.h>
#include <sys/time.h>
#include <chrono>
#define NUM_ITR 4096
__global__ void foo(float * A, int n, float num){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n){
        A[index] = index % (int) num;
    }
}

__global__ void bar(float * A, float * B, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n){
        A[index] = B[index] + A[index];
    }
}
int main(){
    using namespace std::chrono;
    float * A;
    float * B;
    int n = 8192 * 8192;
    cudaMalloc((int**)&A, sizeof(float) * n);
    cudaMalloc((int**)&B, sizeof(float) * n);
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    

    auto start  = high_resolution_clock::now();

    for(int i = 0; i < NUM_ITR; i++){
        foo<<<numBlocks, blockSize>>>(A, n, (float)i);
        bar<<<numBlocks, blockSize>>>(A,B, n);
    }
    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double rt = (double)delta / 1000000;
    printf("latency: %f\n",rt/NUM_ITR);
    cudaFree(A);
    return 0;
}
