#include "../kernel/host_srgemm.h"
#include "../utils/init_mat.h"
#include "../utils/print_mat.h"
#include "../utils/comp_mat.h"

int main() {
  // problem size
  constexpr int M       = 16;
  constexpr int N       = 16;
  constexpr int K       = 1;

  std::cout << "Running host SRGEMM on A = " << M << 'x' << K << " and B = " << K
            << 'x' << N << '\n';

  std::cout << "Allocating and initializing host buffers\n";
  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];

  float *reference_D = new float[M * N];

  rng_init_matrix(A, M * K, 3070);
  rng_init_matrix(B, K * N, 3080);
  rng_init_matrix(C, M * N, 3090);

  //host srgemm
  mmo::host_srgemm<float>(M, N, K, A, M, B, K, C, M, reference_D, true, mmo::plus,mmo::multiplies);
}