#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <float.h>
#include <cusparseLt.h>
#include <chrono>
#include <iostream>
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

// extern "C"
// int cusparseLt_spmm(half * a, half * b, 
//                     half * c, half * d, 
//                     int m, int n, int k, 
//                     float alpha, float beta,
//                     double *kernel_time,
//                     double *total_time){
//     using namespace std::chrono;

//     auto          order = CUSPARSE_ORDER_ROW;
//     auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     auto          type  = CUDA_R_16F;
//     auto          compute_type = CUSPARSE_COMPUTE_16F;

//     bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
//     bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
//     bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
//     auto     num_A_rows     = (isA_transposed) ? k : m;
//     auto     num_A_cols     = (isA_transposed) ? m : k;
//     auto     num_B_rows     = (isB_transposed) ? n : k;
//     auto     num_B_cols     = (isB_transposed) ? k : n;
//     auto     num_C_rows     = m;
//     auto     num_C_cols     = n;
//     unsigned alignment      = 16;
//     auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
//     auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
//     auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
//     auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
//     auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
//     auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
//     auto     A_size         = A_height * lda * sizeof(__half);
//     auto     B_size         = B_height * ldb * sizeof(__half);
//     auto     C_size         = C_height * ldc * sizeof(__half);
//     __half * hA = a;
//     __half * hB = b;
//     __half * hC = c;

//     // Device memory management
//     __half *dA, *dB, *dC, *dD, *dA_compressed;
//     int    *d_valid;
//     CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
//     CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
//     CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
//     CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
//     dD = dC;

//     CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
//     //--------------------------------------------------------------------------
//     cusparseLtHandle_t             handle;
//     cusparseLtMatDescriptor_t      matA, matB, matC;
//     cusparseLtMatmulDescriptor_t   matmul;
//     cusparseLtMatmulAlgSelection_t alg_sel;
//     cusparseLtMatmulPlan_t         plan;
//     cudaStream_t                   stream = nullptr;
//     CHECK_CUSPARSE( cusparseLtInit(&handle) )
//     // matrix descriptor initialization
    
//     CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
//                                             &handle, &matA, num_A_rows,
//                                             num_A_cols, lda, alignment,
//                                             type, order,
//                                             CUSPARSELT_SPARSITY_50_PERCENT) )
//     CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
//                                             &handle, &matB, num_B_rows,
//                                             num_B_cols, ldb, alignment,
//                                             type, order) )
//     CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
//                                             &handle, &matC, num_C_rows,
//                                             num_C_cols, ldc, alignment,
//                                             type, order) )
//     // matmul, algorithm selection, and plan initialization
//     CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
//                                             &handle, &matmul, opA, opB,
//                                             &matA, &matB, &matC, &matC,
//                                             compute_type) )
//     CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
//                                             &handle, &alg_sel, &matmul,
//                                             CUSPARSELT_MATMUL_ALG_DEFAULT) )
//     int alg = 0;
//     CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
//                                             &handle, &alg_sel,
//                                             CUSPARSELT_MATMUL_ALG_CONFIG_ID,
//                                             &alg, sizeof(alg)))
//     size_t workspace_size, compressed_size;
//     CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel,
//                                                  &workspace_size))

//     CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
//                                              workspace_size) )
//     //--------------------------------------------------------------------------
//     // Prune the A matrix (in-place) and check the correcteness
//     auto start_total  = high_resolution_clock::now();
//     CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
//                                          CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
//     CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
//                                               d_valid, stream) )
//     int is_valid;
//     CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid),
//                                 cudaMemcpyDeviceToHost, stream) )
//     CHECK_CUDA( cudaStreamSynchronize(stream) )
//     if (is_valid != 0) {
//         std::printf("!!!! The matrix has been pruned in a wrong way. "
//                     "cusparseLtMatmul will not provide correct results\n");
//         return EXIT_FAILURE;
//     }
//     //--------------------------------------------------------------------------
    
//     // Compress the A matrix
//     CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
//                                                   &compressed_size) )
//     CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

//     CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
//                                             dA_compressed, stream) )
//     //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//     // Search the best kernel
//     void*         d_workspace = nullptr;
//     int           num_streams = 0;
//     cudaStream_t* streams     = nullptr;
//     CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed,
//                                            dB, &beta, dC,dD, d_workspace,
//                                            streams, num_streams) )
//     int alg_id;
//     CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
//                                            &handle, &alg_sel,
//                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
//                                            &alg_id, sizeof(alg_id)) )
//     //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//     // Perform the matrix multiplication
//     auto start  = high_resolution_clock::now();
//     CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
//                                      &beta, dC, dD, d_workspace, streams,
//                                      num_streams) )
//     cudaDeviceSynchronize();
//     auto end    = high_resolution_clock::now();
//     auto delta_kernel = duration_cast<nanoseconds>(end - start).count();
//     auto delta_total = duration_cast<nanoseconds>(end -  start_total).count();
//     *kernel_time = (double)delta_kernel / 1000000.0;
//     *total_time = (double)delta_total / 1000000.0;
//     //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//     // destroy plan and handle
//     CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
//     CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
//     CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
//     CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
//     CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
//     //--------------------------------------------------------------------------
//     // device result check
//     // matrix A has been pruned
//     CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
//     CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )
//     //--------------------------------------------------------------------------
//     // device memory deallocation
//     CHECK_CUDA( cudaFree(dA_compressed) )
//     CHECK_CUDA( cudaFree(dA) )
//     CHECK_CUDA( cudaFree(dB) )
//     CHECK_CUDA( cudaFree(dC) )
//     CHECK_CUDA( cudaFree(d_valid) )

//     return 0;
// }




