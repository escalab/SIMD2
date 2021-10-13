template<typename T>
bool compare_matrix(T * mat_res, T *mat_ref, int m, int n){
  bool f = true;
  for(int i = 0; i < m*n; i++){
    f &= (mat_res[i] == mat_ref[i]);
  }
  return f;
}

template<typename T>
bool compare_matrix_approx(T * mat_res, T *mat_ref, int m, int n, float err_rate){
  bool f = true;
  for(int i = 0; i < m*n; i++){
    if(mat_ref[i] != 0){
      f &= ((abs(mat_res[i] - mat_ref[i]) / mat_ref[i] ) <= err_rate);
    }
    else{
      f &= (mat_res[i] == mat_ref[i]);
    }
    
  }
  return f;
}