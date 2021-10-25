#include "../../../kernel/srgemm.cuh"
#include "../../../kernel/precision.cuh"
#include "../../../kernel/converge.cuh"
#include "../../../utils/check_sum.h"
#include "../../data/graph_gen.h"
#include <sys/time.h>

#include <float.h>
#include <chrono>
#include <string.h>

#define NUM_ITR 20


double gtc_kernel(int * tc, int v) {
    using namespace std::chrono;

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
    auto start  = high_resolution_clock::now();

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
    // printf("itr: %d\n",num_itr);

    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double rt = (double)delta / 1000000;
    cudaMemcpy(tc, out_d, v*v*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
    return rt;
}

int main(int argc, char *argv[]){
    int v;
    // int bound;
    float edge_weight;
    float density;
    // int i,j; // looper
    float *adj_mat; // init adj_mat
    if (argc < 3){
        printf("Usage: ./tc_cuASR num_vertices density\n");
        printf("    number of edges = num_vertices * density\n");
        exit(0);
    }
    if (!strcmp(argv[1], "-f")){
        // add I/O
    }
    else{
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
        printf("failed to malloc dist_tensor\n");
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


    double rt;
    for(int i = 0; i < NUM_ITR; i++){
        rt += gtc_kernel(tc_tensor,v);
    }
    
    // float cs = check_sum<float>(dist_tensor, v*v);
    

    free(adj_mat);
    free(tc_tensor);
    printf("%f\n",rt/NUM_ITR);
    // printf("apsp_cuASR,    check-sum: %f\n",cs);
    return 0;
}