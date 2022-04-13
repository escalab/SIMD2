#include "../../../kernel/srgemm.cuh"
#include "../../../kernel/tensor_srgemm.cuh"
#include "../../../kernel/precision.cuh"
#include "../../../kernel/converge.cuh"
#include "../../../utils/check_sum.h"
#include "../../data/graph_gen.h"

#include <cuda_profiler_api.h>

#include <sys/time.h>

#include <float.h>
#include <chrono>

#define NUM_ITR 20
#define PERFORM

double apsp_kernel(float * adj_mat, float * dist_tensor, int v, int num_itrs, cublasHandle_t cublasHandle){
    using namespace std::chrono;
    float * adj_mat_d; // original graph adj matrix
    float * out_d_delta; // execution result of previous run.
    float * out_d;    // new dist matrix after latest execution

    half * adj_mat_d_fp16; // original graph adj matrix
    half * out_d_delta_fp16; // execution result of previous run.

    int * check_d;
    int * check_h;

    check_h = (int*)malloc(sizeof(int));
    cudaMalloc((int**)&check_d, sizeof(int));
  
    cudaMalloc((float**)&adj_mat_d,v*v*sizeof(float));
    cudaMalloc((float**)&out_d, v*v*sizeof(float));
    cudaMalloc((float**)&out_d_delta, v*v*sizeof(float));

    cudaMalloc((half**)&out_d_delta_fp16, v*v*sizeof(half));
    cudaMalloc((half**)&adj_mat_d_fp16,v*v*sizeof(half));

    cudaMemcpy(adj_mat_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d_delta, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);

    // conversion
    f2h_device(adj_mat_d_fp16, adj_mat_d, v*v);
    f2h_device(out_d_delta_fp16, out_d_delta, v*v);

    auto start  = high_resolution_clock::now();
    for(int i = 0; i < num_itrs; i ++){
        cublas_gemmEx(adj_mat_d_fp16, out_d_delta_fp16, out_d, out_d, v, v, v, 1.0, 1,cublasHandle);
        f2h_device(out_d_delta_fp16, out_d, v*v);
    }
    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double rt = (double)delta / 1000000;

    cudaFree(adj_mat_d);
    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(adj_mat_d_fp16);
    cudaFree(out_d_delta_fp16);
    cudaFree(check_d);
    free(check_h);
    return rt;

}


int apsp_itr_leyz(float * adj_mat, float * dist, int v) {

    float * adj_mat_d; // original graph adj matrix
    float * out_d_delta; // execution result of previous run.
    float * out_d;    // new dist matrix after latest execution

    int * check_d;
    int * check_h;

    check_h = (int*)malloc(sizeof(int));
    cudaMalloc((int**)&check_d, sizeof(int));
  
    cudaMalloc((float**)&adj_mat_d,v*v*sizeof(float));
    cudaMalloc((float**)&out_d, v*v*sizeof(float));
    cudaMalloc((float**)&out_d_delta, v*v*sizeof(float));
  
    //move data (same value initially)
    cudaMemcpy(adj_mat_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d_delta, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);

    bool run = true;
    int num_itr = 0;
    // maximum of 500 iterations of srgemm, wont affect graph with diameter < 500
    while(run && (num_itr < 500)){ 
        num_itr += 1;
        // 1 iteration of minplus srgemm
        int retval = cuasr_minplus_srsgemm(v, v, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        true, nullptr);
        cudaDeviceSynchronize();
        // check convergence
        run = comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }
    cudaMemcpy(dist, out_d, v*v*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(adj_mat_d);
    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
    return num_itr;
}

int apsp_itr(float * adj_mat, float * dist, int v) {

    float * adj_mat_d; // original graph adj matrix
    float * out_d_delta; // execution result of previous run.
    float * out_d;    // new dist matrix after latest execution

    int * check_d;
    int * check_h;

    check_h = (int*)malloc(sizeof(int));
    cudaMalloc((int**)&check_d, sizeof(int));
  
    cudaMalloc((float**)&adj_mat_d,v*v*sizeof(float));
    cudaMalloc((float**)&out_d, v*v*sizeof(float));
    cudaMalloc((float**)&out_d_delta, v*v*sizeof(float));
  
    //move data (same value initially)
    cudaMemcpy(adj_mat_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d_delta, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, adj_mat, v*v*sizeof(float), cudaMemcpyHostToDevice);

    bool run = true;
    int num_itr = 0;
    // maximum of 500 iterations of srgemm, wont affect graph with diameter < 500
    while(run && (num_itr < v)){ 
        num_itr += 1;
        // 1 iteration of minplus srgemm
        int retval = cuasr_minplus_srsgemm(v, v, v, \
                                        adj_mat_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        true, nullptr);
        cudaDeviceSynchronize();
        // check convergence
        run = comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }
    cudaMemcpy(dist, out_d, v*v*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(adj_mat_d);
    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
    return num_itr;
}

int main(int argc, char *argv[]){
    int v;
    int e;
    // int bound;
    float edge_weight;
    float density;
    // int i,j; // looper
    float *adj_mat; // init adj_mat
   
    if (!strcmp(argv[1], "-f")){
        int v1, v2;// value;
        float value;
        std::cin >> v >> e;
        adj_mat = (float*)malloc(v * v * sizeof(float));
        for (int i = 0; i < v*v; i++){
            adj_mat[i] = FLT_MAX;
        }
        for (int i=0; i < e; ++i) {
            std::cin >> v1 >> v2 >> value;
            adj_mat[v1 * v + v2] = -(float)value;
        }
        for(int i = 0; i < v; i++){
            adj_mat[i*v+i] = 0;
        }
        // add I/O
    }
    else{
        if (argc < 4){
            printf("Usage: ./apsp-cuda-v3 num_vertices density edge_weight\n");
            printf("    number of edges = num_vertices * density\n");
            printf("    max edge weight = edge_weight\n");
            exit(0);
        }
        v = atoi(argv[1]);
        density = atof(argv[2]);
        if (density < 0 || density > 1){
            printf("Input density %.2f not within range 0 - 1\n",density);
            exit(0);
        }
        edge_weight = atof(argv[3]);
        // bound = atoi(argv[4]);
        adj_mat = (float*)malloc(v * v * sizeof(float));
        e = rgg_1d(adj_mat, v, density, edge_weight, 7);
    }
  
    
    float * dist_tensor;
    dist_tensor = (float*)calloc(v * v, sizeof(float));
    if (!dist_tensor){
        printf("failed to malloc dist_tensor\n");
        printf("v = %d\n", v);
    }


    int num_itrs = apsp_itr(adj_mat,dist_tensor,v);
    float cs = check_sum<float>(dist_tensor, v*v);
    printf("apsp_cuASR_old,    check-sum: %f\n",cs);
    printf("iters: %d\n",num_itrs);

    num_itrs = apsp_itr_leyz(adj_mat,dist_tensor,v);
    cs = check_sum<float>(dist_tensor, v*v);
    printf("apsp_cuASR_leyz,    check-sum: %f\n",cs);
    printf("iters: %d\n",num_itrs);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    apsp_kernel(adj_mat,dist_tensor,v, num_itrs,cublasHandle);


    double rt = 0.0;
    #ifdef PERFORM
    // for (int i = 0 ; i <  NUM_ITR; i++){
    //     rt += apsp_kernel(adj_mat,dist_tensor,v, num_itrs,cublasHandle);
    // }
    #endif
    
    

    cublasDestroy(cublasHandle);
    free(adj_mat);
    free(dist_tensor);
    // printf("%f %d\n",rt/(double)NUM_ITR, num_itrs);
    // printf("%d\n", num_itrs);
    
    return 0;
}