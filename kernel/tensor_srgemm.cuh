#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <float.h>
/** new wmma simulation kernel -- to support arbitary M,N,K % 16 = 0

  each block only holds 32 threads, that is one warp, each warp performs 16x16x16 gemm.

    ops:
*/
enum op{  min_plus,
          plus_mul,
          max_plus,
          max_min,
          max_mul,
          min_mul,
          min_max,
          or_and,
          minus_square};
    
__device__ float binary_or_and(__half a, __half b, float c){
    int ab, bb, cb;
    ab = (a > (__half) 0) ? 1 : 0;
    bb = (b > (__half) 0) ? 1 : 0;
    cb = (c > 0) ? 1 : 0;
    return (float) (cb || (ab && bb));
    
}

__global__ void mmo_gemm_compute_simple(   half *a, half *b, float *c, float *d, \
                                int m, int n, int k,  \
                                bool do_epilogue,  \
                                enum op opcode){

    // tile ID(block ID)
    int tile_id_y = blockIdx.y;
    int tile_id_x = blockIdx.x;
    // init shared memory
    __shared__ float acc[16][16];
    __shared__ half a_tile[16][16];
    __shared__ half b_tile[16][16];
    __shared__ float c_tile[16][16];
    // thread id within a warp;
    int thread_id_y = threadIdx.y; // 0 - 15
    int thread_id_x = threadIdx.x; // 0 - 1
    int tile_offset = thread_id_x * 8;
    // set acc to 0, each thread set 8
    // load c to c_tile
    float * c_ptr = c + 16 * tile_id_y * n + tile_id_x * 16;
    for(int i = 0; i < 8; i++){
        c_tile[thread_id_y][tile_offset+i] =  c_ptr[thread_id_y * n + tile_offset+i];
        switch (opcode){
            case min_plus:
                acc[thread_id_y][tile_offset+i] = __float2half(FLT_MAX);
                break;
            case plus_mul:
                acc[thread_id_y][tile_offset+i] = 0;
                break;
            case max_plus:
                acc[thread_id_y][tile_offset+i] = __float2half(FLT_MIN);
                break;
            case max_min:
                acc[thread_id_y][tile_offset+i] = __float2half(FLT_MIN);
                break;
            case max_mul:
                acc[thread_id_y][tile_offset+i] = __float2half(FLT_MIN);
                break;
            case min_mul:
                acc[thread_id_y][tile_offset+i] = __float2half(FLT_MAX);
                break;
            case min_max:
                acc[thread_id_y][tile_offset+i] = __float2half(FLT_MAX);
                break;
            case or_and:
                acc[thread_id_y][tile_offset+i] = 0;
                break;
            case minus_square:
                acc[thread_id_y][tile_offset+i] = 0;
                break;
        }
    }
    __syncthreads();

    // loop over K, each time do 16x16x16 mma
    for(int tile_id_k = 0; tile_id_k < k; tile_id_k += 16){
        // find start address of a and b(global address)
        half * a_ptr = a + 16 * tile_id_y * k + tile_id_k;
        half * b_ptr = b + tile_id_k * n + 16 * tile_id_x;
        // half * b_ptr = b + 16 * tile_id_x * k  + tile_id_k;

        // fetch 8 elemnts into a_tile/b_tile
        for(int i = 0; i < 8; i++){
            a_tile[thread_id_y][tile_offset+i] = a_ptr[thread_id_y * k + tile_offset+i];
            b_tile[thread_id_y][tile_offset+i] = b_ptr[thread_id_y * n + tile_offset+i];
            // b_tile[thread_id_y][tile_offset+i] = b_ptr[thread_id_y * k + tile_offset+i];
        }
        __syncthreads();
        // do mma
        for(int i = 0; i < 8; i++){
            for(int j= 0; j < 16; j++){
                float acc_element = acc[thread_id_y][tile_offset+i];
                __half a_element = a_tile[thread_id_y][j];
                __half b_element =  b_tile[j][tile_offset+i];

                switch(opcode){
                  case min_plus:
                    acc[thread_id_y][tile_offset+i] = fmin(acc_element, __half2float(a_element + b_element));
                    break;
                  case plus_mul:
                    acc[thread_id_y][tile_offset+i] = acc_element + __half2float(a_element * b_element);
                    break;
                  case max_plus:
                    acc[thread_id_y][tile_offset+i] = fmax(acc_element, __half2float(a_element + b_element));
                    break;
                  case max_min:
                    acc[thread_id_y][tile_offset+i] = fmax(acc_element, __half2float(__hmin(a_element, b_element)));
                    break;
                  case max_mul:
                    acc[thread_id_y][tile_offset+i] = fmax(acc_element, __half2float(a_element * b_element));
                    break;
                  case min_mul:
                    acc[thread_id_y][tile_offset+i] = fmin(acc_element, __half2float(a_element * b_element));
                    break;
                  case min_max:
                    acc[thread_id_y][tile_offset+i] = fmin(acc_element, __half2float(__hmax(a_element, b_element)));
                    break;
                  case or_and:
                    // acc[thread_id_y][tile_offset+i] = (float) (((int) acc_element ) || ( ((int) __half2float(a_element)) &&  ((int) __half2float(b_element)))); 
                    acc[thread_id_y][tile_offset+i] = binary_or_and(a_element , b_element, acc_element);
                    break;
                  case minus_square:
                    acc[thread_id_y][tile_offset+i] = acc_element + __half2float( (a_element - b_element) * (a_element - b_element) );
                    break;
                }
            }
        }
    }

    //solve beta
    if(do_epilogue){
        for(int i = 0; i < 8; i++){
            float acc_element = acc[thread_id_y][tile_offset+i];
            float c_element = c_tile[thread_id_y][tile_offset+i];
            switch(opcode){
                case min_plus:
                    c_tile[thread_id_y][tile_offset+i] = fmin(acc_element, c_tile[thread_id_y][tile_offset+i]);
                    break;
                case plus_mul:
                    c_tile[thread_id_y][tile_offset+i] = acc_element + c_element;
                    break;
                case max_plus:
                    c_tile[thread_id_y][tile_offset+i] = fmax(acc_element, c_element);
                    break;
                case max_min:
                    c_tile[thread_id_y][tile_offset+i] = fmax(acc_element, c_element);
                    break;
                case max_mul:
                    c_tile[thread_id_y][tile_offset+i] = fmax(acc_element, c_element);
                    break;
                case min_mul:
                    c_tile[thread_id_y][tile_offset+i] = fmin(acc_element, c_element);
                break;
                case min_max:
                    c_tile[thread_id_y][tile_offset+i] = fmin(acc_element, c_element);
                    break;
                case or_and:
                    c_tile[thread_id_y][tile_offset+i] = (int) acc_element || (int) c_element;
                    break;
                case minus_square:
                    c_tile[thread_id_y][tile_offset+i] = acc_element + c_element;
                    break;
            }
        }
    }
    else{
        for(int i = 0; i < 8; i++){
            c_tile[thread_id_y][tile_offset+i] = acc[thread_id_y][tile_offset+i];
        }
    }


    // store back to d
    float * d_ptr = d + 16 * tile_id_y * n + tile_id_x * 16;
    for(int i = 0; i < 8; i++){
        d_ptr[thread_id_y * n + tile_offset+i] = c_tile[thread_id_y][tile_offset+i];
    }
    
} 

extern "C"
int tensor_srgemm(half * a, half * b, float * c, float * d, int m, int n, int k,bool do_epilogue, enum op opcode){
    // printf("[DEBUG] calling mmo_gemm with M: %d, N: %d, K %d\n", m, n ,k);
    if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0){
        return -1;
    }
    dim3 gridDim;
    dim3 blockDim;

    // each block holds 32 (16x2) threads
    // which does 1 16x16x16 gemm
    blockDim.x = 2;
    blockDim.y = 16;

    gridDim.x = (n + 16 - 1) / 16;
    gridDim.y = (m + 16 - 1) / 16;
    // printf(" gridDim.x : %d\n", gridDim.x);
    // printf(" gridDim.y : %d\n", gridDim.y);
    mmo_gemm_compute_simple<<<gridDim, blockDim>>>(a, b, c, d, m, n, k, do_epilogue, opcode);

    return 0;
}

extern "C"
int cublas_sgemm(float * a, float * b, float * c, float * d, int m, int n, int k, float alpha, float beta,cublasHandle_t cublasHandle){
    float alpha_32 = (float)alpha;
    float beta_32 = (float)beta;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
        (int)n, (int)m, (int)k, 
        &alpha_32,
        b,n,
        a,k,
        &beta_32,
        c, n
    );
    return 0;
}

extern "C"
int cublas_gemmEx(half * a, half * b, float * c, float * d, int m, int n, int k, float alpha, float beta,cublasHandle_t cublasHandle){

    float alpha_32 = (float)alpha;
    float beta_32 = (float)beta;
    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
        (int)n, (int)m, (int)k, 
        &alpha_32,
        b, CUDA_R_16F,n,
        a, CUDA_R_16F,k,
        &beta_32,
        c, CUDA_R_32F, n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT 
    );
    return 0;
}



