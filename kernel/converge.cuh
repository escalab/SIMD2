
/**
    Compares prev and cur, 
        racely set check[0] = 1 if any data is different.
**/

__global__ void __comp_update(float * prev, float * cur, int * check, int M, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < M*N){
        if(prev[index] != cur[index]){
            check[0] = 1;
        }
        prev[index] = cur[index];
    }
}

bool comp_update(float * prev, float * cur, int * check, int * check_h, int M, int N){
    int blockSize = 1024;
    int numBlocks = ( M * N + blockSize - 1) / blockSize;

    // reset racer
    check_h[0] = 0;
    cudaMemset(check, 0, sizeof(int));
    __comp_update<<<numBlocks, blockSize>>>(prev, cur,check,M,N);
    cudaDeviceSynchronize();
    cudaMemcpy(check_h, check, sizeof(int),cudaMemcpyDeviceToHost);

    //check racer
    bool res = check_h[0] == 1;
    return res;
}

__global__ void __comp_update2(float * prev, float * cur, int * check, int M, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < M*N){
        if(cur[index] > -1 && cur[index] < 1 ) cur[index] = -FLT_MAX;
        // if((prev[index] - cur[index]) > 0){
        if(prev[index] != cur[index]){
            check[0] = 1;
        }
        prev[index] = cur[index];
        // if(prev[index] > -1 && prev[index] < 1 ) prev[index] = FLT_MAX;
    }
}

bool comp_update2(float * prev, float * cur, int * check, int * check_h, int M, int N){
    int blockSize = 1024;
    int numBlocks = ( M * N + blockSize - 1) / blockSize;

    // reset racer
    check_h[0] = 0;
    cudaMemset(check, 0, sizeof(int));
    __comp_update2<<<numBlocks, blockSize>>>(prev, cur,check,M,N);
    cudaDeviceSynchronize();
    cudaMemcpy(check_h, check, sizeof(int),cudaMemcpyDeviceToHost);

    //check racer
    bool res = check_h[0] == 1;
    return res;
}

__global__ void __comp_update(int * prev, int * cur, int * check, int M, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < M*N){
        if(prev[index] != cur[index]){
            check[0] = 1;
        }
        prev[index] = cur[index];
    }
}

bool comp_update(int * prev, int * cur, int * check, int * check_h, int M, int N){
    int blockSize = 1024;
    int numBlocks = ( M * N + blockSize - 1) / blockSize;

    // reset racer
    check_h[0] = 0;
    cudaMemset(check, 0, sizeof(int));
    __comp_update<<<numBlocks, blockSize>>>(prev, cur,check,M,N);
    cudaDeviceSynchronize();
    cudaMemcpy(check_h, check, sizeof(int),cudaMemcpyDeviceToHost);

    //check racer
    bool res = check_h[0] == 1;
    return res;
}