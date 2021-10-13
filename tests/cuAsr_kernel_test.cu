// unit correctness check for cusar sregmm with defined operators
#include "../kernel/srgemm.cuh"
#include "../utils/init_mat.h"
#include "../utils/print_mat.h"
#include "../utils/comp_mat.h"

#include "../kernel/host_srgemm.h"


/**
  Testers for all srgemm functions.
    Selected kernels leverages cuASR and refernece from simple cpu gemm.
    Keep M,N,K < 512 since reference function is not optimized.
    
    According to https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication

    ** Note **
    To perform GEMM A * B = C, the correct input for device function should be:
                B^T * A^T = C^T, 
      so the device funcion will treat all matirx as column major storage.And  The result stored in C 
      will be C(no transpose needed afterwords.)
*/

bool test_cases(int M, int N, int K, bool do_epilogue, int s1, int s2, int s3){
  bool res = true;
  std::cout << "Running SRGEMM on A = " << M << 'x' << K << " and B = " << K
            << 'x' << N << " do_epilogue = " << do_epilogue <<'\n';

  // std::cout << "Allocating and initializing host/device buffers\n";
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];

  float *reference_D = new float[M * N];
  float *device_D    = new float[M * N];

  rng_init_matrix(A, M * K, s1);
  rng_init_matrix(B, K * N, 3080);
  rng_init_matrix(C, M * N, 3090);

  float *d_A, *d_B, *d_C, *d_D;
  cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  cudaMalloc((void **)&d_C, sizeof(float) * M * N);
  cudaMalloc((void **)&d_D, sizeof(float) * M * N);

  cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  auto retval = 0;
  auto is_correct = true;


//*****************************************************************************
//***************************    MIN_PLUS       *******************************
//*****************************************************************************
  // device srgemm
  // retval = cuasr_minplus_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, do_epilogue,nullptr);
  retval = cuasr_minplus_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
 
  // init reference to inf, otherwise minimum operator would be meaningless
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::max();
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::minimum,mmo::plus);
  std::cout << "Comparing against reference host-side min-plus SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }
  // reset pointers
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval = 0;
//*****************************************************************************
//***************************    PLUS_MUL       *******************************
//*****************************************************************************
  // device srgemm
  retval = cuasr_plusmul_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::plus,mmo::multiplies);
  std::cout << "Comparing against reference host-side plus-mul SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  
  retval = 0;
//*****************************************************************************
//***************************    MAX_PLUS       *******************************
//*****************************************************************************
  // init reference to inf, otherwise minimum operator would be meaningless
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::min();
  }

  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  // device srgemm
  retval = cuasr_maxplus_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::maximum,mmo::plus);
  std::cout << "Comparing against reference host-side max-plus SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);

  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval = 0;
//*****************************************************************************
//***************************    MAX_MIN        *******************************
//*****************************************************************************
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::min();
  }
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  // device srgemm
  retval = cuasr_maxmin_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::maximum,mmo::minimum);
  std::cout << "Comparing against reference host-side max-min SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);

  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval = 0;

//*****************************************************************************
//***************************    MAX_MUL        *******************************
//*****************************************************************************
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::min();
  }
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  // device srgemm
  retval = cuasr_maxmul_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::maximum,mmo::multiplies);
  std::cout << "Comparing against reference host-side max-mul SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval = 0;

//*****************************************************************************
//***************************    MIN_MUL        *******************************
//*****************************************************************************
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::max();
  }
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  // device srgemm
  retval = cuasr_minmul_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::minimum,mmo::multiplies);
  std::cout << "Comparing against reference host-side min-mul SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }

  // reset pointers
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval = 0;
//*****************************************************************************
//***************************    MIN_MAX        *******************************
//*****************************************************************************
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::max();
  }
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  // device srgemm
  retval = cuasr_minmax_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::minimum,mmo::maximum);
  std::cout << "Comparing against reference host-side min-max SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }

  // reset pointers
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
  rng_init_matrix(C, M * N, 3090);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  retval = 0;

//*****************************************************************************
//***************************    OR-AND        ********************************
//*****************************************************************************
  int *A_int = new int[M * K];
  int *B_int = new int[K * N];
  int *C_int = new int[M * N];

  int *reference_D_int = new int[M * N];
  int *device_D_int    = new int[M * N];

  for(int i = 0; i < M*K; i ++){
    A_int[i] = (int) A[i] % 2;
  }

  for(int i = 0; i < N*K; i ++){
    B_int[i] = (int) B[i] % 2;
  }

  for(int i = 0; i < M*N; i ++){
    C_int[i] = (int) C[i] % 2;
  }

  for (int i = 0; i < M * N; i ++){
    reference_D_int[i] = 0;
  }

  int *d_A_int, *d_B_int, *d_C_int, *d_D_int;
  cudaMalloc((void **)&d_A_int, sizeof(int) * M * K);
  cudaMalloc((void **)&d_B_int, sizeof(int) * K * N);
  cudaMalloc((void **)&d_C_int, sizeof(int) * M * N);
  cudaMalloc((void **)&d_D_int, sizeof(int) * M * N);

  cudaMemcpy(d_A_int, A_int, sizeof(int) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_int, B_int, sizeof(int) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C_int, C_int, sizeof(int) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D_int, reference_D_int, sizeof(int) * M * N, cudaMemcpyHostToDevice);

  
  // device srgemm
  retval = cuasr_orand_srsgemm(N, M, K, d_B_int, N, d_A_int, K, d_C_int, N, d_D_int, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D_int, d_D_int, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<int>(M, N, K, A_int, M, B_int, K, C_int, M, reference_D_int, do_epilogue, mmo::bin_or,mmo::bin_and);
  std::cout << "Comparing against reference host-side or-and SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<int>(device_D_int, reference_D_int, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }

  // reset pointers

  retval = 0;
  cudaFree(d_A_int);
  cudaFree(d_B_int);
  cudaFree(d_C_int);
  cudaFree(d_D_int);

  delete [] A_int;
  delete [] B_int;
  delete [] C_int;

//*****************************************************************************
//***************************    PlusMinusSquare        ***********************
//*****************************************************************************
  // device srgemm
  retval = cuasr_plusmiussquare_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N, d_D, do_epilogue,nullptr);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::plus,mmo::minussquare);
  std::cout << "Comparing against reference host-side plus-minussquare SRGEMM : ";
  // compare_matrix_approx does value check approximately
  is_correct =  compare_matrix_approx<float>(device_D, reference_D, M , N, 0.01f);
  // or if you need absolute correctness check
  // auto is_correct = compare_matrix<float>(device_D, reference_D, M , N);
  if (is_correct) {
    std::cout << "PASSED!\n";
  }
  else {
    std::cout << "FAILED!\n";
    res = false;
  }

  // reset pointers
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }

  delete [] A;
  delete [] B;
  delete [] C;
  delete [] reference_D;
  delete [] device_D;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return res;
}


int main() {
  using namespace std::chrono;
  // problem size
  int M,N,K;
  const int num_tests = 25;
  int test_id = 0;
  int passes = 0;
  while (test_id <  num_tests){
    M       = 16;
    N       = 16;
    K       = 4;
    M       = rand() % 256;
    N       = rand() % 256;
    K       = rand() % 256;
    std::cout << "\ntest " << test_id << ":\n";
    bool pass = true;
    pass &= test_cases(M,N,K,false, rand(),rand(),rand());
    pass &= test_cases(M,N,K,true, rand(),rand(),rand());
    test_id++;
    if(pass) passes++;
    else std::cout << "#########################################\n";
  }
  std::cout << passes<<"/"<<num_tests << " tests passed.\n";
  return 0;
}




// print_matrix<float>(A,M, K);
  // std::cout << "\n";
  // print_matrix<float>(B, K, N);
  // std::cout << "\n";
  // print_matrix<float>(device_D, M, N);
  // std::cout << "\n";
  // print_matrix<float>(reference_D, M, N);

 // // store matrix in column major fasion
  // for(int i = 0 ; i < M; i++){
  //   for(int j = 0 ; j < K; j++){
  //     if (i == 0 || j ==0)
  //       A[i*K+j] = 1;
  //     else
  //       A[i*K+j] = 0;
  //   }
  // }


  // for(int i = 0 ; i < K; i++){
  //   for(int j = 0 ; j < N; j++){
  //     if (i == 0 || j ==0)
  //       B[i*N+j] = 1;
  //     else
  //       B[i*N+j] = 0;
  //   }
  // }

  // A[0] = 1;
  // A[1] = 1;
  // A[2] = 1;
  // B[0] = 1;
  // B[1] = 1;
  // B[2] = 1;

  // for(int i = 0 ; i < M; i++){
  //   for(int j = 0 ; j < N; j++){
  //     C[i*K+j] = 1;
  //   }
  // }