#include "../../../kernel/srgemm.cuh"
#include "../../../kernel/precision.cuh"
#include "../../../kernel/converge.cuh"
#include "../../../utils/check_sum.h"
#include "../../data/graph_gen.h"
#include "../../data/from_pbbs.h"
#include <sys/time.h>

#include <float.h>
#include <chrono>

#define NUM_ITR 5



double mst_kernel(float * adj_mat, float * dist, int v) {
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
        int retval = cuasr_minmax_srsgemm(v, v, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d, v, \
                                        out_d_delta, \
                                        false, nullptr);
        cudaDeviceSynchronize();
        // check convergence
        run = comp_update(out_d, out_d_delta, check_d, check_h, v,v);
    }

    cudaDeviceSynchronize();
    auto end    = high_resolution_clock::now();
    auto delta = duration_cast<nanoseconds>(end - start).count();
    double rt = (double)delta / 1000000;
    cudaMemcpy(dist, out_d, v*v*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(adj_mat_d);
    cudaFree(out_d_delta);
    cudaFree(out_d);
    cudaFree(check_d);
    free(check_h);
    printf("num itr: %d\n", num_itr);
    return rt;
}

int main(int argc, char *argv[]){
    int v;
    // int bound;
    float edge_weight;
    float density;
    // int i,j; // looper
    float *adj_mat; // init adj_mat
    // printf("%s\n",argv[1]);
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
  
    float * mst_tensor;
    mst_tensor = (float*)calloc(v * v, sizeof(float));
    if (!mst_tensor){
        printf("failed to malloc mst_tensor\n");
    }
    double rt;
    for(int i = 0; i < NUM_ITR; i++){
        rt += mst_kernel(adj_mat,mst_tensor,v);
    }

    

    free(adj_mat);
    free(mst_tensor);
    printf("%f\n",rt/NUM_ITR);
    // printf("apsp_cuASR,    check-sum: %f\n",cs);
    return 0;
}