#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <float.h>
#include "apsp_parallel_3.h"
#include "apsp_misc.h"

#include <time.h>
#include <sys/time.h>
#include "../../data/graph_gen.h"
#include <string.h>

#define NUM_ITR 10

int main(int argc, char *argv[]){
    int v;
    int e;
    float edge_weight;
    float density;
    int i,j; // looper
    float ** g, **d;
    
    if (!strcmp(argv[1], "-f")){
        int v1, v2;// value;
        float value;
        std::cin >> v >> e;
        g = (float**)malloc(sizeof(float*) * v);
        d = (float**)malloc(sizeof(float*) * v);
        for(int i = 0; i < v; i++){
            g[i] = (float*)malloc(sizeof(float*) * v);
            d[i] = (float*)malloc(sizeof(float*) * v);
        }
        for (int i = 0; i < v; i++){
            for (int j = 0 ; j< v; j++){
                g[i][j] = FLT_MAX;
            }
        }

        for (int i=0; i < e; ++i) {
            std::cin >> v1 >> v2 >> value;
            g[v1][v2] = value;
        }
        for (int i=0; i < v; ++i) {
            g[i][i] = 0;
        }
        
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
        // allocate host memory
        g = (float**)malloc(sizeof(float*) * v);
        d = (float**)malloc(sizeof(float*) * v);
        for(i = 0; i < v; i++){
            g[i] = (float*)malloc(sizeof(float*) * v);
            d[i] = (float*)malloc(sizeof(float*) * v);
        }
        // init graph
        int edge_count = 0;
        e = rgg_2d(g, v, density, edge_weight, 7);
        // std::cout << "Edge_count: " << e << "\n";
    }
    

    // execute apsp-kernel
    unsigned int rt = 0;
    rt = apsp_parallel_3(g,d,v);
    printf("%f\n", rt/NUM_ITR/1000.0);

    // check sum:
    // double checksum = 0;
    // for (i = 0; i < v; i++){
    //     for(j = 0; j < v; j++){
    //         if (d[i][j] <= FLT_MAX && d[i][j] >= FLT_MAX - 5.0){
    //             continue;
    //         }
    //         checksum += d[i][j];
    //     }
        
    // }
    // checksum = sqrt(checksum);
    // printf("apsp-cuda-v3,    check-sum: %f\n",checksum);

    //free host memory
    for(i = 0; i < v; i++){
        free(g[i]);
        free(d[i]);
    }
    free(g);
    free(d);
  

    
}