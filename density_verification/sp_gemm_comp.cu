#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h> // host_device half precision
#include <cublas_v2.h> // cublas apis
#include <cusparse.h> // cusparse apis
#include <cuda.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

#define NUM_ITR 100
// #define VERBOSE
#define SHEET
#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

__host__ void debug_printMat(float * A, int n){
    int i,j;
    printf("\n");
    for(i = 0 ; i < n; i++){
        for(j = 0; j < n; j ++){
            printf("%f ", A[i*n+j]);
        }
        printf("\n");
    }
}
__host__ int init_matrix(int n, float sparsity, float* A){
    int i,j;
    // srand(gettimeofday(NULL,NULL));
    for (i = 0; i < n; i++){
        for(j = 0; j < n; j++){
            if ( ((float)rand()/(float)(RAND_MAX/1000000)) > sparsity){
                A[i*n+j] = (float)rand()/(float)(RAND_MAX/1.0);
            }
        }
    }
    return 1;
}

__host__ int cublas_gemmEx(half * a, half * b, float * c, float * d, int m, int n, int k, float alpha, float beta,cublasHandle_t cublasHandle){

    float alpha_32 = (float)alpha;
    float beta_32 = (float)beta;
    cublasStatus_t cublas_error = cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
        (int)m, (int)n, (int)k, 
        &alpha_32,
        a, CUDA_R_16F, m,
        b, CUDA_R_16F, n,
        &beta_32,
        c, CUDA_R_32F, m,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT 
    );
    if (cublas_error != CUBLAS_STATUS_SUCCESS){
        return -1;
    }
    return 0;
}

__global__ void cuda_float2half(float *floatdata, __half *halfdata, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
		halfdata[index] = (half) floatdata[index];
	}
}
    
int main(){
    
    // srand(10);
    printf("init test...\n");
    // size
    int size_begin = 24576;
    int size_end = 24576;
    // loop over all sizes
    for (int n = size_begin; n <= size_end; n += 1024){
        // default sparsity
        printf("current size: %d\n", n);
        float sparsity = 990000; // all elements are non-zero
        // loop over all sparsity
        while (sparsity <= 990000){
            cudaDeviceReset();
#ifdef VERBOSE
            printf("size: %d X %d, sparsity: %f\n", n, n,sparsity/1000000);
#endif
            // Init matrix
            float * denseMat = (float*)calloc(n * n, sizeof(float));
            init_matrix(n,sparsity,denseMat);
            // debug_printMat(denseMat,n);

            // Sparse format
            int offset = 0;
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    if (denseMat[i*n+j] != 0){
                        offset ++;
                    }
                }
            }
#ifdef VERBOSE
            printf("Non-zero elements: %d\n", offset);
#endif
            int * offsets_h = (int*)malloc( (n+1) *sizeof(int) );
            int * columns_h = (int*)malloc( offset * sizeof(int) );
            float * values_h = (float*)malloc( offset * sizeof(float) );
            offset = 0;
            for(int i = 0; i < n; i++){
                offsets_h[i] = offset;
                for(int j = 0; j < n; j++){
                    if (denseMat[i*n+j] != 0){
                        values_h[offset] = denseMat[i*n+j];
                        columns_h[offset] = j;
                        offset ++;
                    }
                }
            }
            offsets_h[n] = offset;

            // time gemmEX
            // device memory
            float * denseMat_d;
            half * denseMat_fp16_d;
            cudaMalloc((void**)&denseMat_d,        n * n * sizeof(float));
            cudaMalloc((void**)&denseMat_fp16_d,   n * n * sizeof(half));
            cudaMemcpy(denseMat_d,   denseMat,    n * n * sizeof(float),    cudaMemcpyHostToDevice);
            // float to half conversion
            int blockSize = 1024;
            int numBlocks = ( n*n + blockSize - 1) / blockSize;
	          checkKernelErrors((cuda_float2half<<<numBlocks, blockSize>>>(denseMat_d, denseMat_fp16_d, n*n)));

            // init cublas
            cublasHandle_t cublasHandle;
            cublasCreate(&cublasHandle);

            // warm-up run
            int retval;
            retval = cublas_gemmEx(denseMat_fp16_d, denseMat_fp16_d, denseMat_d, denseMat_d, n, n, n, 1.0, 0.0, cublasHandle);
            if (retval == -1){
                printf("GemmEX failed at %d\n",__LINE__);
            }

            // cudaMemcpy(denseMat,   denseMat_d,    n * n * sizeof(float),    cudaMemcpyDeviceToHost);
            // debug_printMat(denseMat,n);

            //timers
            struct timeval ts;
            struct timeval te;
            gettimeofday(&ts, NULL);

            // timing steps
            for(int i = 0; i < NUM_ITR; i++){
                retval = cublas_gemmEx(denseMat_fp16_d, denseMat_fp16_d, denseMat_d, denseMat_d, n, n, n, 1.0, 0.0, cublasHandle);
                
            }
            cudaDeviceSynchronize();
            gettimeofday(&te, NULL);
            double elapsed_time = te.tv_sec - ts.tv_sec;
            double gemmEX_time = (elapsed_time + (te.tv_usec - ts.tv_usec) / 1000000.) / NUM_ITR;
            // printf("GEMMEX TIME: %f ms\n", gemmEX_time*1000);
          // exit(0);

            cudaMemcpy(denseMat,   denseMat_d,    n * n * sizeof(float),    cudaMemcpyDeviceToHost);
            // debug_printMat(denseMat,n);

            // free resoures
            cublasDestroy(cublasHandle);
            cudaFree(denseMat_d);
            cudaFree(denseMat_fp16_d);


            // time spgemm
            cudaDeviceReset();
            const int A_nnz      = offset;
            float               alpha       = 1.0f;
            float               beta        = 0.0f;
            cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
            cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
            cudaDataType        computeType = CUDA_R_32F;

            int   *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns, *dC_csrOffsets, *dC_columns;
            float *dA_values, *dB_values, *dC_values;

            // allocate A
            CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets, (n+1) * sizeof(int)) )
            CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int)) )
            CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float)) )

            CHECK_CUDA( cudaMalloc((void**) &dB_csrOffsets, (n+1) * sizeof(int)) )
            CHECK_CUDA( cudaMalloc((void**) &dB_columns, A_nnz * sizeof(int)) )
            CHECK_CUDA( cudaMalloc((void**) &dB_values,  A_nnz * sizeof(float)) )

            // allocate C offsets
            CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,(n+1) * sizeof(int)) )

            // copy A
            CHECK_CUDA( cudaMemcpy(dA_csrOffsets, offsets_h,(n+1) * sizeof(int),cudaMemcpyHostToDevice) )
            CHECK_CUDA( cudaMemcpy(dA_columns, columns_h, A_nnz * sizeof(int),cudaMemcpyHostToDevice) )
            CHECK_CUDA( cudaMemcpy(dA_values, values_h, A_nnz * sizeof(float), cudaMemcpyHostToDevice) )

            // copy B
            CHECK_CUDA( cudaMemcpy(dB_csrOffsets, offsets_h,(n+1) * sizeof(int),cudaMemcpyHostToDevice) )
            CHECK_CUDA( cudaMemcpy(dB_columns, columns_h, A_nnz * sizeof(int),cudaMemcpyHostToDevice) )
            CHECK_CUDA( cudaMemcpy(dB_values, values_h, A_nnz * sizeof(float), cudaMemcpyHostToDevice) )
            
            // CUSPARSE APIs
            cusparseHandle_t     handle = NULL;
            cusparseSpMatDescr_t matA, matB, matC;
            void*  dBuffer1    = NULL, *dBuffer2   = NULL;
            size_t bufferSize1 = 0,    bufferSize2 = 0;
            CHECK_CUSPARSE( cusparseCreate(&handle) )
            
            // Create sparse matrix A,B,C in CSR format
            CHECK_CUSPARSE( cusparseCreateCsr(&matA, n, n, A_nnz,
                dA_csrOffsets, dA_columns, dA_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

            CHECK_CUSPARSE( cusparseCreateCsr(&matB, n, n, A_nnz,
                dA_csrOffsets, dA_columns, dA_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

            CHECK_CUSPARSE( cusparseCreateCsr(&matC, n, n, 0,
                NULL, NULL, NULL,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

            //--------------------------------------------------------------------------
            // SpGEMM Computation
            gettimeofday(&ts, NULL);
            int64_t C_num_rows1, C_num_cols1, C_nnz1;
            
            cusparseSpGEMMDescr_t spgemmDesc;
            CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

            CHECK_CUSPARSE(
                cusparseSpGEMM_workEstimation(handle, opA, opB,
                                            &alpha, matA, matA, &beta, matC,
                                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc, &bufferSize1, NULL) )

            CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )

            // inspect the matrices A and B to understand the memory requiremnent for the next step
            CHECK_CUSPARSE(
                cusparseSpGEMM_workEstimation(handle, opA, opB,
                                            &alpha, matA, matA, &beta, matC,
                                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc, &bufferSize1, dBuffer1) )
        
            // ask bufferSize2 bytes for external memory
            CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                            &alpha, matA, matA, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, NULL) )
            CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

            cudaDeviceSynchronize();

            for(int i = 0; i < NUM_ITR; i++){
                // compute the intermediate product of A * B
                CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                &alpha, matA, matA, &beta, matC,
                computeType, CUSPARSE_SPGEMM_DEFAULT,
                spgemmDesc, &bufferSize2, dBuffer2) )
                // get matrix C non-zero entries C_nnz1
            }
            
            cudaDeviceSynchronize();
            CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                &C_nnz1) )
            // allocate matrix C
            CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
            CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )
            // update matC with the new pointers
            CHECK_CUSPARSE( cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

            // copy the final products to the matrix C
            CHECK_CUSPARSE( cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matA, &beta, matC,
                                computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

            
            // destroy matrix/vector descriptors
            CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
           
            gettimeofday(&te, NULL);
            elapsed_time = te.tv_sec - ts.tv_sec;
            double spgemmEX_time = (elapsed_time + (te.tv_usec - ts.tv_usec) / 1000000.) / NUM_ITR;
            // printf("SPGEMM TIME: %f ms\n", spgemmEX_time*1000);
                
            //--------------------------------------------------------------------------
            CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
            CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
            CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
            CHECK_CUSPARSE( cusparseDestroy(handle) )

            int * c_offset_h = (int*)malloc((n+1) * sizeof(int));
            int * c_columns_h = (int*)malloc(C_nnz1 * sizeof(int));
            float * c_values_h = (float*)malloc(C_nnz1 * sizeof(float));
            CHECK_CUDA( cudaMemcpy(c_offset_h, dC_csrOffsets, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost) )
            CHECK_CUDA( cudaMemcpy(c_columns_h, dC_columns, C_nnz1 * sizeof(int), cudaMemcpyDeviceToHost) )
            CHECK_CUDA( cudaMemcpy(c_values_h, dC_values, C_nnz1 * sizeof(float), cudaMemcpyDeviceToHost) )

            // check ans
            // printf("C-NNZ: %ld\n",C_nnz1);
            // for(int i = 0; i < n+1; i ++){
            //     printf("%d ", c_offset_h[i]);
            // }
            // printf("\n");
            // for(int i = 0; i < C_nnz1; i ++){
            //     printf("%d ", c_columns_h[i]);
            // }
            // printf("\n");
            // for(int i = 0; i < C_nnz1; i ++){
            //     printf("%f ", c_values_h[i]);
            // }
            // printf("\n");

            float * sparseMat = (float*)calloc( n * n, sizeof(float));
            for (int i = 0; i < n; i++){
                int begin = c_offset_h[i];
                int end = c_offset_h[i+1];

                for(int j = begin; j < end; j++ ){
                    sparseMat[i*n+c_columns_h[j]] = c_values_h[j];
                }
            }
            // debug_printMat(denseMat,n);
            // debug_printMat(sparseMat,n);

            // check correctness
            int f = 0;
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    if ( (abs(denseMat[i*n+j] - sparseMat[i*n+j]) / sparseMat[i*n+j]) > 0.05) {
                        f = 1;
                        break;
                    }
                }
                if(f) break;
            }
#ifdef VERBOSE
            printf("GEMMEX TIME: %f ms\n", gemmEX_time*1000);
            printf("SPGEMM TIME: %f ms\n", spgemmEX_time*1000);
#endif

#ifdef SHEET
            printf("%f %f\n", gemmEX_time*1000, spgemmEX_time*1000);
#endif
            // if(!f){
            //     printf("Test Passed\n");
            // }
            // else{
            //     printf("Test Failed\n");
            // }
            
            // increase sparsity, free source matrix
            sparsity += 1000;
            free(denseMat);
        }

    }
}
    
