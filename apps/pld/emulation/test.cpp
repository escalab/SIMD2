#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "knncuda.h"

#include "../../../utils/print_mat.h"

void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    // srand(time(NULL));
    srand(1024);

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10.0 * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10.0 * (float)(rand() / (double)RAND_MAX);
    }
}

bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *, double*),
          const char *  name,
          int           nb_iterations) {

    // Parameters
    const float precision    = 0.1f; // distance error max
    const float min_accuracy = 0.9f; // percentage of correct values required

    
    
    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // reshape data if mxu/emmulation kernel is called
    float * ref_transpose;
    if(name == "knn_mxu" || name == "knn_tensor"){
        // printf("transpose data\n");
        ref_transpose = (float*)malloc(ref_nb * dim * sizeof(float));
        for(int i = 0; i < ref_nb; i++){
            for(int j = 0; j < dim; j++){
            ref_transpose[i*dim + j] = ref[j*ref_nb + i];
            }
        }
        ref = ref_transpose;
    }
    
    // Display k-NN function name
    if  (nb_iterations > 1){
        printf("- %-25s : ", name);
    }
    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);
    double elapsed_time_kernel = 0;

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index, &elapsed_time_kernel)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
    

    // Verify both precisions and indexes of the k-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<query_nb*k; ++i) {
        // printf("check distances: %f/%f     index: %d/%d\n", test_knn_dist[i],gt_knn_dist[i], test_knn_index[i],gt_knn_index[i]);
        if (fabs(test_knn_dist[i] - gt_knn_dist[i]) <= precision) {
            nb_correct_precisions++;
        }
        if (test_knn_index[i] == gt_knn_index[i]) {
            nb_correct_indexes++;
        }
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) query_nb * k);
    float index_accuracy     = nb_correct_indexes    / ((float) query_nb * k);

    if  (nb_iterations > 1){
        std::cout << (elapsed_time_kernel/nb_iterations)*1000 << "\n";
    }

    // Free memory
    free(test_knn_dist);
    free(test_knn_index);
    if(name == "knn_mxu" || name == "knn_tensor"){
      free(ref_transpose);
    }

    return true;
}


/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */
int main(int argc, char **argv) {

    if (argc < 5){
        printf("usage: ./test ref_nb query_nb dim k\n");
        exit(0);
    }
    const int ref_nb   = atoi(argv[1]);
    const int query_nb = atoi(argv[2]);
    const int dim      = atoi(argv[3]);
    const int k        = atoi(argv[4]);

    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    // Compute the ground truth k-NN distances and indexes for each query point
    double t;
    if (!knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &t)) {
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Test all k-NN functions
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_mxu,          "knn_mxu",       1);
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_mxu,          "knn_mxu",       100);
    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
