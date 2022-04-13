#include "../../../kernel/srgemm.cuh"
#include "../../../kernel/precision.cuh"
#include "../../../kernel/tensor_srgemm.cuh"
#include "../../../kernel/converge.cuh"
#include "../../../utils/check_sum.h"
#include "../../data/graph_gen.h"
#include <sys/time.h>

#include <float.h>
#include <chrono>

#define NUM_ITR 10
double apsp_time(float * adj_mat, float * dist_tensor, int v, int num_itrs, cublasHandle_t cublasHandle){
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
        // cudaDeviceSynchronize();
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

void apsp_approx(float * adj_mat, float * dist, int v, int num_itrs){
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

    int num_itr = 0;
    // maximum of 500 iterations of srgemm, wont affect graph with diameter < 500
    while(num_itr < num_itrs){ 
        num_itr += 1;
        // 1 iteration of minplus srgemm
        int retval = cuasr_minplus_srsgemm(v, v, v, \
                                        adj_mat_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        true, nullptr);
        cudaDeviceSynchronize();
        // only update
        comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(dist, out_d, v*v*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(adj_mat_d);
    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
}

int apsp_kernel(float * adj_mat, float * dist, int v) {
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
                                        adj_mat_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        true, nullptr);
        cudaDeviceSynchronize();
        // check convergence
        run = comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }
    cudaDeviceSynchronize();
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
    float edge_weight;
    float density;
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
            adj_mat[v1 * v + v2] = value;
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
        int edge_count = rgg_1d(adj_mat, v, density, edge_weight, 7);
    }
  
    
    float * dist_tensor_ans;
    float * dist_tensor;
    dist_tensor = (float*)calloc(v * v, sizeof(float));
    dist_tensor_ans = (float*)calloc(v * v, sizeof(float));
    if (!dist_tensor){
        printf("failed to malloc dist_tensor\n");
    }
    if (!dist_tensor_ans){
        printf("failed to malloc dist_tensor_ans\n");
    }
    int max_iter;
    max_iter = apsp_kernel(adj_mat,dist_tensor_ans,v);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    for (int i = 1; i < max_iter; i++){
        double rt = 0.0;
        for (int j = 0 ; j <  NUM_ITR; j++){
            rt += apsp_time(adj_mat,dist_tensor,v, i,cublasHandle);
        }
        apsp_approx(adj_mat, dist_tensor, v, i);
        double rmse = 0;
        rmse = get_rmse(dist_tensor_ans, dist_tensor, v*v);

        printf("Iteration bound = %d, RMSE: %lf, time: %f\n", i, rmse, rt/(double)NUM_ITR);

    }
    
    
    cublasDestroy(cublasHandle);
    free(adj_mat);
    free(dist_tensor);
    free(dist_tensor_ans);
    // printf("apsp_cuASR,    check-sum: %f\n",cs);
    return 0;
}