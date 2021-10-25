#include "../../../kernel/srgemm.cuh"
#include "../../../kernel/tensor_srgemm.cuh"
#include "../../../kernel/precision.cuh"
#include "../../../kernel/converge.cuh"
#include "../../../utils/check_sum.h"
#include "../../data/graph_gen.h"
#include "../../data/from_pbbs.h"
#include <sys/time.h>

#include <float.h>
#include <chrono>

#define NUM_ITR 20

double gtc_kernel(float * adj_mat, int v, int num_itrs, cublasHandle_t cublasHandle){
    using namespace std::chrono;
    float * out_d_delta; // execution result of previous run.
    float * out_d;    // new dist matrix after latest execution

    half * adj_mat_d_fp16; // original graph adj matrix
    half * out_d_delta_fp16; // execution result of previous run.

    int * check_d;
    int * check_h;

    check_h = (int*)malloc(sizeof(int));
    cudaMalloc((int**)&check_d, sizeof(int));
  

    cudaMalloc((float**)&out_d, v*v*sizeof(float));
    cudaMalloc((float**)&out_d_delta, v*v*sizeof(float));

    cudaMalloc((half**)&out_d_delta_fp16, v*v*sizeof(half));
    cudaMalloc((half**)&adj_mat_d_fp16,v*v*sizeof(half));

    cudaMemcpy(out_d_delta, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);

    // conversion
    f2h_device(out_d_delta_fp16, out_d_delta, v*v);

    auto start  = high_resolution_clock::now();
    for(int i = 0; i < num_itrs; i ++){
        cublas_gemmEx(out_d_delta_fp16, out_d_delta_fp16, out_d, out_d, v, v, v, 1.0, 1,cublasHandle);
        f2h_device(out_d_delta_fp16, out_d, v*v);
        // cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double rt = (double)delta / 1000000;

    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(out_d_delta_fp16);
    cudaFree(check_d);
    free(check_h);
    return rt;
}


int gtc_itr(int * tc, int v) {

    int * out_d_delta; // execution result of previous run.
    int * out_d;    // new dist matrix after latest execution

    int * check_d;
    int * check_h;

    check_h = (int*)malloc(sizeof(int));
    cudaMalloc((int**)&check_d, sizeof(int));
  
    cudaMalloc((int**)&out_d, v*v*sizeof(int));
    cudaMalloc((int**)&out_d_delta, v*v*sizeof(int));
  
    //move data (same value initially)
    cudaMemcpy(out_d_delta, tc, v*v*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, tc, v*v*sizeof(int), cudaMemcpyHostToDevice);

    bool run = true;
    int num_itr = 0;
    // maximum of 500 iterations of srgemm, wont affect graph with diameter < 500
    while(run && (num_itr < 500)){ 
        num_itr += 1;
        // 1 iteration of minplus srgemm
        int retval = cuasr_orand_srsgemm(v, v, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        true, nullptr);
        cudaDeviceSynchronize();
        // check convergence
        run = comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }
    cudaMemcpy(tc, out_d, v*v*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
    return num_itr;
}

int main(int argc, char *argv[]){
    int v;
    // int bound;
    // float edge_weight;
    float density;
    // int i,j; // looper
    float *adj_mat; // init adj_mat
    
    if (!strcmp(argv[1], "-f")){
        // printf("read from file\n");
        v = 1 + count_from_pbbs(argv[2]);
        // printf("num vertices: %d\n", v);
        adj_mat = (float*)malloc(v * v * sizeof(float));
        read_from_pbbs(argv[2],v,adj_mat);
        int edge_count = 0;
        for (int i = 0; i < v*v; i++){
            if (adj_mat[i] < FLT_MAX - 100) edge_count ++;
        }
        // printf("num edges: %d\n", edge_count);

    }
    else{
        if (argc < 3){
            printf("Usage: ./apsp-cuda-v3 num_vertices density\n");
            printf("    number of edges = num_vertices * density\n");
            exit(0);
        }
        v = atoi(argv[1]);
        density = atof(argv[2]);
        if (density < 0 || density > 1){
            printf("Input density %.2f not within range 0 - 1\n",density);
            exit(0);
        }
        // bound = atoi(argv[4]);
        adj_mat = (float*)malloc(v * v * sizeof(float));
        int edge_count = rgg_1d_directed(adj_mat, v, density, 10, 7);
    }
  
    
    int * tc_tensor;
    tc_tensor = (int*)calloc(v * v, sizeof(int));
    if (!tc_tensor){
        printf("failed to malloc mst_tensor\n");
    }
    for(int i = 0 ; i <  v; i ++){
        for (int j = 0; j < v; j++){
            if (i == j) tc_tensor[i * v + j] = 1;
            else{
                if ( adj_mat[i * v + j] < (FLT_MAX - 100) ){
                    tc_tensor[i * v + j] = 1;
                }
                else{
                    tc_tensor[i * v + j] = 0;
                }
            }
        }
    }


    int num_itrs = gtc_itr(tc_tensor,v);
    // printf("iters: %d\n",num_itrs);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    gtc_kernel(adj_mat,v, num_itrs,cublasHandle);


    double rt = 0.0;
    for (int i = 0 ; i <  NUM_ITR; i++){
        rt += gtc_kernel(adj_mat,v, num_itrs,cublasHandle);
    }
    
    

    cublasDestroy(cublasHandle);
    free(adj_mat);
    free(tc_tensor);
    printf("%f\n",rt/NUM_ITR);
    // printf("%d\n", num_iters);
    // printf("apsp_cuASR,    check-sum: %f\n",cs);
    return 0;
}