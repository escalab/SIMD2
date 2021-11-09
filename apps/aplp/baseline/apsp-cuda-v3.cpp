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

#include "../../../utils/check_sum.h"

#define NUM_ITR 5

int main(int argc, char *argv[]){
    int v,e;
    float ** g, **d;
    int v1, v2;// value;
    float value;
    std:: cin >> v >> e;

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
        g[v1][v2] = -value;
    }
    // for (int i=0; i < v; ++i) {
    //     g[i][i] = FLT_MAX;
    // }

    // execute apsp-kernel
    unsigned int rt = 0;
    rt = apsp_parallel_3(g,d,v);
    printf("%f\n", rt/NUM_ITR/1000.0);

    // check sum:
    // int valid_count = 0;
    // double checksum = 0;
    // for (int i = 0; i < v; i++){
    //     for(int j = 0; j < v; j++){
    //         if (abs(d[i][j]) < (FLT_MAX-100)){
    //             checksum += -d[i][j];
    //             valid_count ++;
    //             // printf("%.3f\n", d[i][j]); 
    //         }
    //     }
        
    // }
    // std::cout << valid_count << " entry valid\n";
    // checksum = sqrt(checksum);
    // printf("apsp-cuda-v3,    check-sum: %f\n",checksum);

    //free host memory
    for(int i = 0; i < v; i++){
        free(g[i]);
        free(d[i]);
    }
    free(g);
    free(d);
  

    
}