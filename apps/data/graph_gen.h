/**
 Random Graph Generator
**/
#include <float.h>

int rgg_2d(float ** g, int v, float density, float edge_weight, int seed){
  int edge_count = 0;
  srand(seed);
  for (int i = 0; i < v; i++){
      for (int j = 0; j < v; j++){
          if (i == j){
              g[i][j] = 0.0;
              continue;
          }
          if ( j > i){
              float weight = (float)rand()/(float)(RAND_MAX/edge_weight);
              float has_edge = (float)rand()/(float)(RAND_MAX/1);
              if (has_edge < density){
                  g[i][j] = weight;
                  edge_count += 1;
              }
              else{
                  g[i][j] = FLT_MAX;
              }
              g[j][i] = g[i][j];
          }
      }
  }
  return edge_count;
}


int rgg_1d(float * g, int v, float density, float edge_weight, int seed){
  int edge_count = 0;
  srand(seed);
  for (int i = 0; i < v; i++){
      for (int j = 0; j < v; j++){
          if (i == j){
              g[i * v + j] = 0.0;
              continue;
          }
          if ( j > i){
              float weight = (float)rand()/(float)(RAND_MAX/edge_weight);
              float has_edge = (float)rand()/(float)(RAND_MAX/1);
              if (has_edge < density){
                  g[i * v + j] = weight;
                  edge_count += 1;
              }
              else{
                  g[i * v + j] = FLT_MAX;
              }
              g[j* v + i] = g[i * v + j];
          }
      }
  }
  return edge_count;
}

int rgg_1d_directed(float *g, int v, float density, float edge_weight,  int seed){
    int edge_count = 0;
    srand(seed);
    int i,j;
    for(i = 0; i < v ;i ++){
        for(j = 0; j < v; j++){
            if (i == j){
              g[i * v + j] = 0.0;
              continue;
            }
            float weight = (float)rand()/(float)(RAND_MAX/edge_weight);
            float has_edge = (float)rand()/(float)(RAND_MAX/1);
            if (has_edge < density){
                g[i * v + j] = weight;
                edge_count += 1;
            }
            else{
                g[i * v + j] = FLT_MAX;
            }
        }
    }
    return edge_count;
}

    