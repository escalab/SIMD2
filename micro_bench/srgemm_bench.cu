#include "../kernel/tensor_srgemm.cuh"
#include "../kernel/srgemm.cuh"
#include "../kernel/precision.cuh"
#include "../kernel/host_srgemm.h"

#include "../utils/init_mat.h"
#include "../utils/print_mat.h"
#include "../utils/comp_mat.h"


#include <chrono>

#define NUM_ITR 1


void bench_case(int M, int N, int K, int s1, int s2, int s3){

  std::cout << "Running microbench on A = " << M << 'x' << K << " and B = " << K
            << 'x' << N << '\n';

  using namespace std::chrono;

  // Host tensors
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];
  float *device_D  = new float[M * N];
  for (int i = 0; i < M * N; i ++){
    device_D[i] = 0;
  }

  // init host tensors
  rng_init_matrix(A, M * K, s1);
  rng_init_matrix(B, K * N, 3080);
  rng_init_matrix(C, M * N, 3090);

  // device tensors
  float *d_A, *d_B, *d_C, *d_D;
  half *d_A_fp16, *d_B_fp16;
  

  cudaMalloc((void **)&d_A, sizeof(float) * M * K);
  cudaMalloc((void **)&d_B, sizeof(float) * K * N);
  cudaMalloc((void **)&d_C, sizeof(float) * M * N);
  cudaMalloc((void **)&d_D, sizeof(float) * M * N);

  cudaMalloc((void **)&d_A_fp16, sizeof(half) * M * K);
  cudaMalloc((void **)&d_B_fp16, sizeof(half) * K * N);

  // copy data
  cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, device_D, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // conversion
  f2h_device(d_A_fp16, d_A, M*K);
  f2h_device(d_B_fp16, d_B, N*K);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);


  double cuASR_min_plus = 0;
  double cuASR_plus_mul = 0;
  double cuASR_max_plus = 0;
  double cuASR_max_min = 0;
  double cuASR_max_mul = 0;
  double cuASR_min_mul = 0;
  double cuASR_min_max = 0;
  double cuASR_or_and = 0;
  double cuASR_minus_square = 0;
  double __cublas_sgemm = 0;
  double __cublas_gemmEx = 0;

  auto start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_minplus_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  auto end    = high_resolution_clock::now();
  auto delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_min_plus = delta / NUM_ITR;


  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_plusmul_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_plus_mul = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_maxplus_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_max_plus = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_maxmin_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_max_min = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_maxmul_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_max_mul = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_minmul_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_min_mul = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_minmax_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_min_max = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_plusmiussquare_srsgemm(N, M, K, d_B, N, d_A, K, d_C, N,d_D, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_minus_square = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cublas_sgemm(d_A, d_B, d_D, d_D, M, N, K, 1.0, 0,cublasHandle);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  __cublas_sgemm = delta / NUM_ITR;

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cublas_gemmEx(d_A_fp16, d_B_fp16, d_D, d_D, M, N, K, 1.0, 0,cublasHandle);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  __cublas_gemmEx = delta / NUM_ITR;


  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_A_fp16);
  cudaFree(d_B_fp16);
  cudaFree(d_C);
  cudaFree(d_D);
  cublasDestroy(cublasHandle);

  int *A_int = new int[M * K];
  int *B_int = new int[K * N];
  int *C_int = new int[M * N];
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
    device_D_int[i] = 0;
  }

  delete [] A;
  delete [] B;
  delete [] C;
  delete [] device_D;

  int *d_A_int, *d_B_int, *d_C_int, *d_D_int;
  cudaMalloc((void **)&d_A_int, sizeof(int) * M * K);
  cudaMalloc((void **)&d_B_int, sizeof(int) * K * N);
  cudaMalloc((void **)&d_C_int, sizeof(int) * M * N);
  cudaMalloc((void **)&d_D_int, sizeof(int) * M * N);

  cudaMemcpy(d_A_int, A_int, sizeof(int) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_int, B_int, sizeof(int) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C_int, C_int, sizeof(int) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D_int, device_D_int, sizeof(int) * M * N, cudaMemcpyHostToDevice);

  start  = high_resolution_clock::now();
  for(int i = 0; i < NUM_ITR; i++){
    cuasr_orand_srsgemm(N, M, K, d_B_int, N, d_A_int, K, d_C_int, N,d_D_int, false, nullptr);
  }
  cudaDeviceSynchronize();
  end    = high_resolution_clock::now();
  delta = duration_cast<nanoseconds>(end - start).count();
  cuASR_or_and = delta / NUM_ITR;

  cudaFree(d_A_int);
  cudaFree(d_B_int);
  cudaFree(d_C_int);
  cudaFree(d_D_int);

  delete [] A_int;
  delete [] B_int;
  delete [] C_int;




  std::cout << "cuASR_min_plus: " << cuASR_min_plus / 1000.0 << " ms\n";
  std::cout << "cuASR_plus_mul: " << cuASR_plus_mul / 1000.0 << " ms\n";
  std::cout << "cuASR_max_plus: " << cuASR_max_plus / 1000.0 << " ms\n";
  std::cout << "cuASR_max_min: " << cuASR_max_min / 1000.0 << " ms\n";
  std::cout << "cuASR_max_mul: " << cuASR_max_mul / 1000.0 << " ms\n";
  std::cout << "cuASR_min_mul: " << cuASR_min_mul / 1000.0 << " ms\n";
  std::cout << "cuASR_min_max: " << cuASR_min_max / 1000.0 << " ms\n";
  std::cout << "cuASR_or_and: " << cuASR_or_and / 1000.0 << " ms\n";
  std::cout << "cuASR_minus_square: " <<  cuASR_minus_square / 1000.0 << " ms\n";
  std::cout << "cublas_sgemm: " <<  __cublas_sgemm / 1000.0 << " ms\n";
  std::cout << "cublas_gemmEx: " <<  __cublas_gemmEx / 1000.0 << " ms\n";


}

int main(){
  // bench_case(1024, 1024, 1024, rand(), rand(), rand());
  // bench_case(2048, 2048, 2048, rand(), rand(), rand());
  bench_case(4096, 4096, 4096, rand(), rand(), rand());
 // bench_case(8192, 8192, 8192, rand(), rand(), rand());
//   bench_case(16384, 16384, 16384, rand(), rand(), rand());

  //bench_case(2048, 1024, 512, rand(), rand(), rand());
  //bench_case(4096, 2048, 1024, rand(), rand(), rand());
  //bench_case(8192, 4096, 2048, rand(), rand(), rand());
  //bench_case(16384, 8192, 4096, rand(), rand(), rand());
  //bench_case(32768, 16384, 4096, rand(), rand(), rand());
  //bench_case(65536, 32768, 16384, rand(), rand(), rand());
  return 0;
}
