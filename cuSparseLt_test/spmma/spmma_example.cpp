/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>

#define NUM_ITR 50

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

constexpr int EXIT_UNSUPPORTED = 2;

int test(int M, int N, int K, float density) {
    using namespace std::chrono;
    // New data generator
    __half * a_h = new __half[M * K];
    __half * b_h = new __half[K * N];
    __half * c_h = new __half[M * N];
    float dens = density;

    for (int i = 0; i < M * K; i++){
        float pass = (float)rand()/(float)(RAND_MAX/1);
        if (pass < dens) a_h[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
        // else a_h[i] = (__half) 0;
        
    }   
    for (int i = 0; i < K * N; i++){
        float pass = (float)rand()/(float)(RAND_MAX/1);
        if (pass < dens) b_h[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
        // else b_h[i] = (__half) 0;
    }
    // Adding hgemm here
    __half *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, sizeof(__half) * M * K);
    cudaMalloc((void **)&b_d, sizeof(__half) * K * N);
    cudaMalloc((void **)&c_d, sizeof(__half) * M * N);
    cudaMemcpy(a_d, a_h, sizeof(__half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(__half) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h, sizeof(__half) * M * N, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    __half alpha_16 = (__half)1.0f;
    __half beta_16 = (__half)0.0f;
    //warm-up
    cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
        (int)N, (int)M, (int)K, 
        &alpha_16,
        b_d,N,
        a_d,M,
        &beta_16,
        c_d, K
    );
    cudaDeviceSynchronize();

    auto start  = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITR; i++){
        cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
            (int)N, (int)M, (int)K, 
            &alpha_16,
            b_d,N,
            a_d,M,
            &beta_16,
            c_d, K
        );
    }
    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double hgemm_time = delta / NUM_ITR;
    // std::printf("hgemm time: %lf\n", hgemm_time/1000.0);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cublasDestroy(cublasHandle);

    //
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    // Host problem definition, row-major order
    // constexpr int m     = M; // bigger sizes may require dynamic allocations
    // constexpr int n     = N; // bigger sizes may require dynamic allocations
    // constexpr int k     = K; // bigger sizes may require dynamic allocations
    int m = M;
    int n = N;
    int k = K;
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type  = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_16F;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size         = A_height * lda * sizeof(__half);
    auto     B_size         = B_height * ldb * sizeof(__half);
    auto     C_size         = C_height * ldc * sizeof(__half);
    __half * hA = a_h;
    __half * hB = b_h;
    __half * hC = c_h;
    // for (int i = 0; i < m * k; i++)
    //     hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
    // for (int i = 0; i < k * n; i++)
    //     hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel,
                                                 &workspace_size))

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correcteness
    
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    
    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha, dA_compressed,
                                           dB, &beta, dC,dD, d_workspace,
                                           streams, num_streams) )
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &alg_id, sizeof(alg_id)) )
    // printf("best alg: %d\n", alg_id);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    start  = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITR; i++){
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    }
    cudaDeviceSynchronize();
    end    = high_resolution_clock::now();
    delta = duration_cast<nanoseconds>(end - start).count();
    double spmm_time = delta / NUM_ITR;
    std::printf("%d %f %lf %lf\n", M, dens, hgemm_time/1000.0,spmm_time/1000.0);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    // bool A_std_layout = (is_rowmajor != isA_transposed);
    // bool B_std_layout = (is_rowmajor != isB_transposed);
    // // host computation
    // float hC_result[m * n];
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         float sum  = 0.0f;
    //         for (int k1 = 0; k1 < k; k1++) {
    //             auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
    //             auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
    //             sum      += static_cast<float>(hA[posA]) *  // [i][k]
    //                         static_cast<float>(hB[posB]);   // [k][j]
    //         }
    //         auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
    //         hC_result[posC] = sum;  // [i][j]
    //     }
    // }
    // // host-device comparison
    // int correct = 1;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
    //         auto device_value = static_cast<float>(hC[pos]);
    //         auto host_value   = hC_result[pos];
    //         if (device_value != host_value) {
    //             // direct floating point comparison is not reliable
    //             std::printf("(%d, %d):\t%f vs. %f\n",
    //                         i, j, host_value, device_value);
    //             correct = 0;
    //             break;
    //         }
    //     }
    // }
    // if (correct)
    //     std::printf("spmma_example test PASSED\n");
    // else
    //     std::printf("spmma_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    delete [] a_h;
    delete [] b_h;
    delete [] c_h;
    return EXIT_SUCCESS;
}

int main(){
    float dens[8] = {0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999};
    for (int j = 0; j < 8; j++){
        for (int i = 512; i <= 16384; i *= 2){
            test(i,i,i,dens[j]);
        }
    }
}
