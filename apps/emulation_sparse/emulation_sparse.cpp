#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>
#include <iostream>
#include <cfloat>

#define NUM_ITR 1

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

int cusparseLt_spmm(half * a, half * b, 
                    half * c, half * d, 
                    int m, int n, int k, 
                    float alpha, float beta,
                    double *kernel_time,
                    double *total_time){
    using namespace std::chrono;

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
    __half * hA = a;
    __half * hB = b;
    __half * hC = c;

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
    auto start_total  = high_resolution_clock::now();
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
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    auto start  = high_resolution_clock::now();
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta_kernel = duration_cast<nanoseconds>(end - start).count();
    auto delta_total = duration_cast<nanoseconds>(end -  start_total).count();
    *kernel_time = (double)delta_kernel / 1000000.0;
    *total_time = (double)delta_total / 1000000.0;
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
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )

    return 0;
}

void float2half_cpu(float *data_fp32, __half * data_fp16, int n){
    for (int i = 0; i < n; i++){
        data_fp16[i] =  static_cast<__half>(static_cast<float>(data_fp32[i]));
    }
}

void half2float_cpu(float *data_fp32, __half * data_fp16, int n){
    for (int i = 0; i < n; i++){
        data_fp32[i] =  static_cast<float>(data_fp16[i]);
    }
}

void apsp_kernel(float * adj_mat, float * dist_tensor, 
    int v, int num_itrs,
    double *kernel_time,
    double *total_time){
    using namespace std::chrono;
    __half * adj_mat_fp16 = (half*)malloc(v * v * sizeof(half));
    __half * out = (half*)calloc(v * v, sizeof(half));
    __half * out_delta = (half*)calloc(v * v, sizeof(half));
    float2half_cpu(adj_mat, adj_mat_fp16, v * v);

    double kernel_time_delta = 0;
    double total_time_delta = 0;
    for(int i = 0; i < num_itrs; i ++){
        cusparseLt_spmm(adj_mat_fp16, out_delta, out, out,
             v, v, v, 1.0, 1, 
             &kernel_time_delta, &total_time_delta);
        *kernel_time += kernel_time_delta;
        *total_time += total_time_delta;
    }
}

int main(int argc, char *argv[]){
    int v;
    int e;
    int num_itrs = atoi(argv[1]);
    float edge_weight;
    float density;
    // int i,j; // looper
    float *adj_mat; // init adj_mat
   
    int v1, v2;// value;
    float value;
    std::cin >> v >> e;
    adj_mat = (float*)malloc(v * v * sizeof(float));
    for (int i = 0; i < v*v; i++){
        adj_mat[i] = FLT_MAX;
    }
    for (int i=0; i < e; ++i) {
        std::cin >> v1 >> v2 >> value;
        adj_mat[v1 * v + v2] = (float) value;
    }
    for(int i = 0; i < v; i++){
        adj_mat[i*v+i] = 0;
    }
  
    
    float * dist_tensor;
    dist_tensor = (float*)calloc(v * v, sizeof(float));
    if (!dist_tensor){
        printf("failed to malloc dist_tensor\n");
        printf("v = %d\n", v);
    }

    double total_time = 0;
    double kernel_time = 0;
    apsp_kernel(adj_mat,dist_tensor,v, num_itrs,&kernel_time, &total_time);


    for (int i = 0 ; i <  NUM_ITR; i++){
        apsp_kernel(adj_mat,dist_tensor,v, num_itrs, &kernel_time, &total_time);
    }

    free(adj_mat);
    free(dist_tensor);
    printf("%f %f %d\n",total_time/(double)NUM_ITR, kernel_time/(double)NUM_ITR, num_itrs);
    return 0;
}