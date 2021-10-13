#include "../kernel/tensor_srgemm.cuh"
#include "../kernel/precision.cuh"
#include "../kernel/host_srgemm.h"

#include "../utils/init_mat.h"
#include "../utils/print_mat.h"
#include "../utils/comp_mat.h"

#include <float.h>

/**
    Testers for all tensor srgemm kernels. (Also include gemm correctness check for cublas api)
*/

bool test_cases(int M, int N, int K, bool do_epilogue, int s1, int s2, int s3){
  bool res = true;
  std::cout << "Running SRGEMM on A = " << M << 'x' << K << " and B = " << K
            << 'x' << N << " do_epilogue = " << do_epilogue <<'\n';

  // std::cout << "Allocating and initializing host/device buffers\n";
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];

  half *A_fp16 = new half[M*K];
  half *B_fp16 = new half[M*K];


  float *reference_D = new float[M * N];
  float *device_D    = new float[M * N];

  rng_init_matrix(A, M * K, s1);
  rng_init_matrix(B, K * N, s2);
  rng_init_matrix(C, M * N, s3);
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
  float *d_A, *d_B, *d_C, *d_D;
  half *d_A_fp16, *d_B_fp16;

  cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  cudaMalloc((void **)&d_C, sizeof(float) * M * N);
  cudaMalloc((void **)&d_D, sizeof(float) * M * N);

  cudaMalloc((void **)&d_A_fp16, sizeof(half) * M * K);
  cudaMalloc((void **)&d_B_fp16, sizeof(half) * K * N);

  cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  f2h_device(d_A_fp16, d_A, M*K);
  f2h_device(d_B_fp16, d_B, N*K);

  
  auto retval = 0;
  auto is_correct = true;

//*****************************************************************************
//***************************    PLUS_MUL       *******************************
//*****************************************************************************
  // device srgemm
  retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,plus_mul);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }

 
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::plus,mmo::multiplies);
  std::cout << "Compare device with host-side plus-mul SRGEMM : ";
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

  // test cublas api

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  float beta = (do_epilogue) ? 1 : 0;

  // test sgemm
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
    

  
  cudaMemcpy(d_D, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cublas_sgemm(d_A, d_B, d_D, d_D, M, N, K, 1.0, beta,cublasHandle);

  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);

  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::plus,mmo::multiplies);
  std::cout << "Compare device with host-side plus-mul SRGEMM (sgemm) : ";
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


  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
  cudaMemcpy(d_D, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cublas_gemmEx(d_A_fp16, d_B_fp16, d_D, d_D, M, N, K, 1.0, beta,cublasHandle);

  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::plus,mmo::multiplies);
  std::cout << "Compare device with host-side plus-mul SRGEMM (gemmEx) : ";
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

  cublasDestroy(cublasHandle);


  // print_matrix<float>(A,M, K);
  // std::cout << "\n";
  // print_matrix<float>(B, K, N);
  // std::cout << "\n";
  // // print_matrix<float>(C, M, N);
  // // std::cout << "\n";
  // print_matrix<float>(device_D, M, N);
  // std::cout << "\n";
  // print_matrix<float>(reference_D, M, N);

  
  

  // test gemmex
  for (int i = 0; i < M * N; i ++){
    reference_D[i] = 0;
  }
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  
  

//*****************************************************************************
//***************************    MIN_PLUS       *******************************
//*****************************************************************************

  for (int i = 0; i < M * N; i ++){
    reference_D[i] = std::numeric_limits<float>::max();
  }
  cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // device srgemm
  retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,min_plus);
  cudaDeviceSynchronize();
  if (retval) {
    std::cout << "Error code " << retval << '\n';
  }

 
  cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::minimum,mmo::plus);
  std::cout << "Compare device with host-side min-plus SRGEMM : ";
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

//*****************************************************************************
//***************************    MAX_PLUS       *******************************
//*****************************************************************************

for (int i = 0; i < M * N; i ++){
  reference_D[i] = std::numeric_limits<float>::min();
}
cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,max_plus);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm
mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::maximum,mmo::plus);
std::cout << "Compare device with host-side min-plus SRGEMM : ";
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

//*****************************************************************************
//***************************    MAX_MIN        *******************************
//*****************************************************************************

for (int i = 0; i < M * N; i ++){
  reference_D[i] = std::numeric_limits<float>::min();
}
cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,max_min);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm
mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::maximum,mmo::minimum);
std::cout << "Compare device with host-side max-min SRGEMM : ";
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

//*****************************************************************************
//***************************    MAX_MUL        *******************************
//*****************************************************************************

for (int i = 0; i < M * N; i ++){
  reference_D[i] = std::numeric_limits<float>::min();
}
cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,max_mul);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm
mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::maximum,mmo::multiplies);
std::cout << "Compare device with host-side max-mul SRGEMM : ";
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

//*****************************************************************************
//***************************    MIN_MUL        *******************************
//*****************************************************************************

// for (int i = 0; i < M * N; i ++){
//   reference_D[i] = std::numeric_limits<float>::min();
// }
// cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,min_mul);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm
mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::minimum,mmo::multiplies);
std::cout << "Compare device with host-side min-mul SRGEMM : ";
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

//*****************************************************************************
//***************************    MIN_MAX        *******************************
//*****************************************************************************

for (int i = 0; i < M * N; i ++){
  reference_D[i] = std::numeric_limits<float>::max();
}
cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,min_max);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm
mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::minimum,mmo::maximum);
std::cout << "Compare device with host-side min-max SRGEMM : ";
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

//*****************************************************************************
//***************************    OR_AND        *******************************
//*****************************************************************************

int *A_int = new int[M * K];
int *B_int = new int[K * N];
int *C_int = new int[M * N];
int *reference_D_int = new int[M * N];
int *device_D_int    = new int[M * N];
for (int i = 0; i < M * N; i ++){
  reference_D[i] = 0;
  reference_D_int[i] = 0;
}
for(int i = 0; i < M*K; i ++){
  A_int[i] = (int) A[i] % 2;
  A[i] = A_int[i];
}

for(int i = 0; i < N*K; i ++){
  B_int[i] = (int) B[i] % 2;
  B[i] = B_int[i];
}

for(int i = 0; i < M*N; i ++){
  C_int[i] = (int) C[i] % 2;
  C[i] = C_int[i];
}

cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

f2h_device(d_A_fp16, d_A, M*K);
f2h_device(d_B_fp16, d_B, N*K);


// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,or_and);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm



mmo::host_srgemm<int>(M, N, K, A_int, M, B_int, K, C_int, M, reference_D_int, do_epilogue, mmo::bin_or,mmo::bin_and);

for (int i = 0; i < M * N; i ++){
  reference_D[i] = reference_D_int[i];
}
std::cout << "Compare device with host-side or-and SRGEMM : ";
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

//  print_matrix<float>(A,M, K);
//   std::cout << "\n";
//   print_matrix<float>(B, K, N);
//   std::cout << "\n";
//   // print_matrix<float>(C, M, N);
//   // std::cout << "\n";
//   print_matrix<float>(device_D, M, N);
//   std::cout << "\n";
//   print_matrix<float>(reference_D, M, N);

//*****************************************************************************
//***************************    PLUS-MINUSSQUARE     *************************
//*****************************************************************************

for (int i = 0; i < M * N; i ++){
  reference_D[i] = 0;
}
cudaMemcpy(d_D, reference_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

// device srgemm
retval = tensor_srgemm(d_A_fp16,d_B_fp16,d_C,d_D, M,N,K,do_epilogue,minus_square);
cudaDeviceSynchronize();
if (retval) {
  std::cout << "Error code " << retval << '\n';
}


cudaMemcpy(device_D, d_D, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
//host srgemm
mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, do_epilogue, mmo::plus,mmo::minussquare);
std::cout << "Compare device with host-side plus-minussquare SRGEMM : ";
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

  // print_matrix<float>(A,M, K);
  // std::cout << "\n";
  // print_matrix<float>(B, K, N);
  // std::cout << "\n";
  // print_matrix<float>(C, M, N);
  // std::cout << "\n";
  // print_matrix<float>(device_D, M, N);
  // std::cout << "\n";
  // print_matrix<float>(reference_D, M, N);
  


  delete [] A;
  delete [] B;
  delete [] C;
  delete [] reference_D;
  delete [] device_D;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_A_fp16);
  cudaFree(d_B_fp16);
  cudaFree(d_C);
  cudaFree(d_D);

  return res;
  
}

int main() {
  // problem size
  int M,N,K;
  const int num_tests = 30;
  int test_id = 0;
  int passes = 0;
  srand(time(NULL));
  while (test_id <  num_tests){
    M       = 16;
    N       = 32;
    K       = 16;
    M       = (rand() % 32 + 1) * 16;
    N       = (rand() % 32 + 1) * 16;
    K       = (rand() % 32 + 1) * 16;
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
  // // print_matrix<float>(C, M, N);
  // // std::cout << "\n";
  // print_matrix<float>(device_D, M, N);
  // std::cout << "\n";
  // print_matrix<float>(reference_D, M, N);