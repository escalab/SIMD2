/**
    CuBool baseline performe graph transitive clousre, modified to:
    1. initmatrix in dense format and convet to sparse formate
    2. added I/O // TODO
    3. timing
**/

/************************************************/
/* Evaluate transitive closure for some graph G */
/************************************************/

/* Actual cubool C API */
#include <cubool/cubool.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <float.h>
#include <string.h>

#include <chrono>

#include "../../data/graph_gen.h"
#include "../../../utils/print_mat.h"

#define NUM_ITR 10

/* Macro to check result of the function call */
#define CHECK(f) { cuBool_Status s = f; if (s != CUBOOL_STATUS_SUCCESS) return s; }

int main(int argc, char *argv[]) {

    using namespace std::chrono;

    int v;
    float density;
    float *adj_mat;

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
        adj_mat = (float*)malloc(v * v * sizeof(float));
    }
    int edge_count = rgg_1d_directed(adj_mat, v, density, 10, 7);

    std::cout << "edge_count: " << edge_count << "\n";

    // print_matrix<float>(adj_mat,v,v);

    
    // cubool sparse structures
    cuBool_Matrix A;
    cuBool_Matrix TC;

    /* System may not provide Cuda compatible device */
    CHECK(cuBool_Initialize(CUBOOL_HINT_NO));

    /* Adjacency matrix in sparse format  */

    // copy from input
    cuBool_Index n = v;
    cuBool_Index e = edge_count;

    // cuBool_Index n = 4;
    // cuBool_Index e = 5;
    cuBool_Index * rows;
    cuBool_Index * cols;
    rows = (cuBool_Index *) malloc(e * sizeof(cuBool_Index));
    cols = (cuBool_Index *) malloc(e * sizeof(cuBool_Index));
    // cuBool_Index rows[] = { 0, 0, 1, 2, 3 };
    // cuBool_Index cols[] = { 1, 2, 2, 3, 2 };

    //print matrix
    // for(int i = 0 ; i <  v; i ++){
    //     for (int j = 0; j < v; j++){
    //         if ( adj_mat[i * v + j] < (FLT_MAX - 100) ){
    //             printf("%.2f ",adj_mat[i * v + j]);
    //         }
    //         else printf("inf ");
    //     }
    //     printf("\n");
    // }


    // fill rows and cols
    int edge_id = 0;
    for(int i = 0 ; i <  v; i ++){
        for (int j = 0; j < v; j++){
            if (i == j) continue;
            if ( adj_mat[i * v + j] < (FLT_MAX - 100) ){
                rows[edge_id] = i;
                cols[edge_id] = j;
                edge_id ++;
                if (edge_id >= edge_count) break;
            }
        }
        if (edge_id >= edge_count) break;
    }

    // std::cout << "rows: ";
    // for(int i = 0 ; i < e; i++){
    //     std::cout << rows[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "cols: ";
    // for(int i = 0 ; i < e; i++){
    //     std::cout << cols[i] << " ";
    // }
    // std::cout << "\n";

    /* Create matrix */
    CHECK(cuBool_Matrix_New(&A, n, n));
    /* Fill the data */
    CHECK(cuBool_Matrix_Build(A, rows, cols, e, CUBOOL_HINT_VALUES_SORTED));

     // free host memory
    free(rows);
    free(cols);
    free(adj_mat);


    
    

    double rt = 0;
    cuBool_Index total = 0;
    cuBool_Index current;
    int itr_count = 0;

    for(int i = 0; i < NUM_ITR; i ++){
        /* Create result matrix from source as copy */
        CHECK(cuBool_Matrix_Duplicate(A, &TC));

        /* Query current number on non-zero elements */
        total = 0;
        CHECK(cuBool_Matrix_Nvals(TC, &current));

        auto start  = high_resolution_clock::now();

        /* Loop while values are added */
        itr_count = 0;
        while (current != total) {
            total = current;
            /** Transitive closure step */
            CHECK(cuBool_MxM(TC, TC, TC, CUBOOL_HINT_ACCUMULATE));
            CHECK(cuBool_Matrix_Nvals(TC, &current));
            itr_count ++;
        }

        auto end    = high_resolution_clock::now();
        auto delta = duration_cast<nanoseconds>(end - start).count();
        rt += (double)delta / 1000000;
        
    }

    std::cout << rt/NUM_ITR << "\n";
    // std::cout << "Itr: " << itr_count << "\n";

    /** Get result */
    cuBool_Index * tc_rows; 
    cuBool_Index * tc_cols;
    tc_rows = (cuBool_Index *) malloc(total * sizeof(cuBool_Index));
    tc_cols = (cuBool_Index *) malloc(total * sizeof(cuBool_Index));
    CHECK(cuBool_Matrix_ExtractPairs(TC, tc_rows, tc_cols, &total));


    /* Output result size */
    // printf("Nnz(tc)=%lli\n", (unsigned long long) total);

    // for (cuBool_Index i = 0; i < total; i++)
    //     printf("(%u,%u) ", tc_rows[i], tc_cols[i]);

    /* Release resources */
    CHECK(cuBool_Matrix_Free(A));
    CHECK(cuBool_Matrix_Free(TC));
    free(tc_rows);
    free(tc_cols);

    /* Release library */
    return cuBool_Finalize() != CUBOOL_STATUS_SUCCESS;
}