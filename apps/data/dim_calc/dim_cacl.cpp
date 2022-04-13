#include <cstdio>
#include <iostream>
#include <limits>

using namespace std;

// #define NOT_CONNECTED std::numeric_limits<int>::max();

int main(){

    int v;
    int e;
    long long *distance; // init adj_mat
    int v1, v2;
    float value;

    // int NOT_CONNECTED  = -1;

    std::cin >> v >> e;
    distance = (long long*)malloc(v * v * sizeof(long long));
    for (int i = 0; i < v*v; i++){
        distance[i] = (long long)-1;
    }
    for (int i = 0; i < v; i++){
        distance[i*v+i] = (long long) 0;
    }
    for (int i=0; i < e; ++i) {
        std::cin >> v1 >> v2 >> value;
        distance[v1 * v + v2] = (long long) 1;
    }
    int nodesCount = v;
    int m = e;
    printf("G: v = %d e = %d\n", nodesCount,m);

    //print mat
    // for (int i = 0; i < v; i++){
    //     for (int j = 0; j < v; j++){
    //         printf("%d ", distance[i*v+j]);
    //     }
    //     printf("\n");  
    // }
    //Floyd-Warshall
    for (int k=0;k<v;k++){
        for (int i=0;i<v;i++){
            for (int j=0;j<v;j++){
                if (distance[i*v+k] != -1 && distance[k*v+j] != -1){
                    long long new_dist = distance[i*v+k] + distance[k*v+j];
                    // printf("%d\n",new_dist);
                    if (distance[i*v+j] == -1){
                        distance[i*v+j] = new_dist;
                    }
                    else{
                        if (distance[i*v+j] > new_dist) distance[i*v+j] = new_dist;
                    }
                }
            }
        }
    }
    printf("END OF F-W\n");
    //print mat
    // for (int i = 0; i < 16; i++){
    //     for (int j = 0; j < 16; j++){
    //         printf("%lld ", distance[i*v+j]);
    //     }
    //     printf("\n");  
    // }

    long long diameter=0;

    //look for the most distant pair
    for (int i=0;i<v;i++){
        for (int j=0;j<v;j++){
            if (distance[i * v + j] != -1){
                if (diameter < distance[i * v + j]) diameter = distance[i * v + j];
                // printf("%d %d %d\n", i, j, diameter);
            }
        }
    }

    

    int path_count = 0;
    for (int i = 1; i < v*v; i++){
        if (distance[i] != -1) path_count++;
    }
    printf("Path count: %d\n", path_count);
    printf("Diameter: %lld\n", diameter);
    free(distance);
    return 0;
}