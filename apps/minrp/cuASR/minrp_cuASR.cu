#include "../../../kernel/srgemm.cuh"
#include "../../../kernel/precision.cuh"
#include "../../../kernel/converge.cuh"
#include "../../../utils/check_sum.h"
#include "../../../utils/print_mat.h"
#include "../../data/graph_gen.h"
#include <sys/time.h>

#include <float.h>
#include <chrono>
#include <iostream>
#include <unistd.h>

#define NUM_ITR 10



double minrp_kernel(float * adj_mat, float * dist, int v) {
    using namespace std::chrono;

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
    auto start  = high_resolution_clock::now();

    // maximum of 500 iterations of srgemm, wont affect graph with diameter < 500
    while(run && (num_itr < 500)){ 
        num_itr += 1;
        // 1 iteration of minplus srgemm
        int retval = cuasr_minmul_srsgemm(v, v, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        true, nullptr);
        cudaDeviceSynchronize();
        // check convergence
        run = comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }

    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double rt = (double)delta / 1000000;
    cudaMemcpy(dist, out_d, v*v*sizeof(float), cudaMemcpyDeviceToHost);

    // printf("num_itr: %d\n", num_itr);
    cudaFree(adj_mat_d);
    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
    return rt;
}

int main(int argc, char *argv[]){
    int v;
    int e;
    // int bound;
    // float edge_weight = 10;
    // float density = 0.01;
    // int i,j; // looper
    float *adj_mat; // init adj_mat
    
    // if (strcmp(argv[1], "-g")){
    int v1, v2;
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

    float * dist_tensor;
    dist_tensor = (float*)calloc(v * v, sizeof(float));
    if (!dist_tensor){
        printf("failed to malloc dist_tensor\n");
    }
    double rt;
    for(int i = 0; i < NUM_ITR; i++){
        rt += minrp_kernel(adj_mat,dist_tensor,v);
    }
    
    // float cs = check_sum<float>(dist_tensor, v*v);
    
    // print_matrix<float>(dist_tensor,v,v);

    free(adj_mat);
    free(dist_tensor);
    printf("%f\n",rt/NUM_ITR);
    // printf("apsp_cuASR,    check-sum: %f\n",cs);
    return 0;
}