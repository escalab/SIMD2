#include <iostream>

template<typename T>
void print_matrix(T * data, int m, int n){
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      std::cout << data[i*n+j] << " ";
    }
    std::cout << std::endl;
  }
}