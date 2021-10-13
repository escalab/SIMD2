#include <chrono>
#include <iostream>
#include <random>

#include "../cuASR/include/cuasr/gemm/device/default_srgemm_configuration.h"
#include "../cuASR/include/cuasr/gemm/device/srgemm.h"
#include "../cuASR/include/cuasr/functional.h"
#include "../cuASR/tools/include/cuasr/reference/srgemm/host_srgemm.h"

/**
  CuASR kernels supports all needed operators:
    1. min_plus
    2. plus_mul
    3. max_plus
    4. max_min
    5. max_mul
    6. min_mul
    7. min_max
    8. or_and
    9. minus_square
**/
//*****************************************************************************
//***************************    MIN_PLUS       *******************************
//*****************************************************************************

auto cuasr_minplus_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
using OperatorClass    = cutlass::arch::OpClassSimt;
using SmArch           = cutlass::arch::Sm50;
using AdditionOp       = cuasr::minimum<float>;
using MultiplicationOp = cuasr::plus<float>;

using TropicalConfig = typename cuasr::gemm::device::DefaultSemiRingConfiguration<
    float, float, float, float, OperatorClass, //
    AdditionOp, MultiplicationOp, SmArch>;

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;

using cuASR_MinPlus_SGEMM = cuasr::gemm::device::Srgemm<
    AdditionOp,       // Thread level SemiRing operator
    MultiplicationOp, // Thread level SemiRing operator
    float,            // element type of A
    ColumnMajor,      // layout of A
    float,            // element type of B
    ColumnMajor,         // layout of B
    float,            // element t  ype of C
    ColumnMajor,         // layout of C
    float             // element type of D
    >;

float alpha = MultiplicationOp::Identity;
float beta
    = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

// construct kernel arguments struct
cuASR_MinPlus_SGEMM::Arguments args(
    { M, N, K },    // Problem dimensions
    { A, lda },     // Tensor-ref for source matrix A
    { B, ldb },     // Tensor-ref for source matrix B
    { C, ldc },     // Tensor-ref for source matrix C
    { D, ldc },     // Tensor-ref for destination matrix D
    { alpha, beta } //
);

// launch SRGEMM kernel
cuASR_MinPlus_SGEMM minplus_gemm;
cutlass::Status status = minplus_gemm(args, nullptr, stream);
return static_cast<int>(status);
}

//*****************************************************************************
//***************************    PLUS_MUL       *******************************
//*****************************************************************************
auto cuasr_plusmul_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::plus<float>;
  using MultiplicationOp = cuasr::multiplies<float>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      ColumnMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  float beta = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }

//*****************************************************************************
//***************************    MAX_PLUS       *******************************
//*****************************************************************************
auto cuasr_maxplus_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::maximum<float>;
  using MultiplicationOp = cuasr::plus<float>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      ColumnMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  // float beta = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;
  // ** Note **
  //  MultiplicationOp::Annihilator which should be -inf will cause unkown bug here, 
  //  use small neg number to resolve the issue
  float beta = do_epilogue_min ? MultiplicationOp::Identity : std::numeric_limits<float>::min();

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }

//*****************************************************************************
//***************************    MAX_MIN        *******************************
//*****************************************************************************
auto cuasr_maxmin_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::maximum<float>;
  using MultiplicationOp = cuasr::minimum<float>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      ColumnMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  float beta = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }
//*****************************************************************************
//***************************    MAX_MUL        *******************************
//*****************************************************************************
auto cuasr_maxmul_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::maximum<float>;
  using MultiplicationOp = cuasr::multiplies<float>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      ColumnMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  float beta = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }

//*****************************************************************************
//***************************    MIN_MUL        *******************************
//*****************************************************************************
auto cuasr_minmul_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::minimum<float>;
  using MultiplicationOp = cuasr::multiplies<float>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      ColumnMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  float beta = do_epilogue_min ? MultiplicationOp::Identity : std::numeric_limits<float>::max();

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }


//*****************************************************************************
//***************************    MIN_MAX        *******************************
//*****************************************************************************
auto cuasr_minmax_srsgemm(
  int M,
  int N,
  int K,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float *C,
  int ldc,
  float *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::minimum<float>;
  using MultiplicationOp = cuasr::maximum<float>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      float,            // element type of A
      ColumnMajor,      // layout of A
      float,            // element type of B
      ColumnMajor,         // layout of B
      float,            // element t  ype of C
      ColumnMajor,         // layout of C
      float             // element type of D
      >;

  float alpha = MultiplicationOp::Identity;
  float beta = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }


//*****************************************************************************
//***************************    OR-AND        ********************************
//*****************************************************************************
auto cuasr_orand_srsgemm(
  int M,
  int N,
  int K,
  int const *A,
  int lda,
  int const *B,
  int ldb,
  int *C,
  int ldc,
  int *D,
  bool do_epilogue_min,
  cudaStream_t stream = nullptr) -> int {
// compile time configuration of this srgemm kernel using OperatorClass
  using OperatorClass    = cutlass::arch::OpClassSimt;
  using SmArch           = cutlass::arch::Sm80;

  using AdditionOp       = cuasr::binary_or<int>;
  using MultiplicationOp = cuasr::binary_and<int>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor    = cutlass::layout::RowMajor;

  using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,       // Thread level SemiRing operator
      MultiplicationOp,  // Thread level SemiRing operator
      int,            // element type of A
      ColumnMajor,      // layout of A
      int,            // element type of B
      ColumnMajor,         // layout of B
      int,            // element t  ype of C
      ColumnMajor,         // layout of C
      int             // element type of D
      >;

      int alpha = MultiplicationOp::Identity;
      int beta = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;

  // construct kernel arguments struct
  cuASR_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } //
  );

  // launch SRGEMM kernel
  cuASR_SGEMM srgemm;
  cutlass::Status status = srgemm(args, nullptr, stream);
  return static_cast<int>(status);
  }


//*****************************************************************************
//***************************    PlusMinusSquare        ***********************
//*****************************************************************************
namespace {
  template <typename T, int N = 1>
  struct minussquare {
    static T constexpr Identity = static_cast<T>(1);
    // static T constexpr Annihilator = std::numeric_limits<T>::max();
    static T constexpr Annihilator = static_cast<T>(0);
  
    // expose base scalar operator
    __host__ __device__
    T operator()(T lhs, T const &rhs) const {
      lhs = (lhs - rhs) * (lhs - rhs);
      // lhs += rhs;
      return lhs;
    }
  
    __host__ __device__
    cutlass::Array<T, N>
    operator()(cutlass::Array<T, N> const &lhs, cutlass::Array<T, N> const &rhs) const {
      cutlass::Array<T, N> result;
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        result[i] = this->operator()(lhs[i], rhs[i]);
      }
      return result;
    }
  
    __host__ __device__
    cutlass::Array<T, N>
    operator()(cutlass::Array<T, N> const &lhs, T const &scalar) const {
      cutlass::Array<T, N> result;
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        // result[i] = this->operator()(lhs[i], scalar);
        result[i] = lhs[i];
      }
      return result;
    }
  
    __host__ __device__
    cutlass::Array<T, N>
    operator()(T const &scalar, cutlass::Array<T, N> const &rhs) const {
      cutlass::Array<T, N> result;
      #pragma unroll
      for (int i = 0; i < N; ++i) {
        // result[i] = this->operator()(scalar, rhs[i]);
        result[i] = rhs[i];
      }
      return result;
    }
  };
  } // namespace:operator

  auto cuasr_plusmiussquare_srsgemm(
    int M,
    int N,
    int K,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float *C,
    int ldc,
    float *D,
    bool do_epilogue_min,
    cudaStream_t stream = nullptr) -> int {
  // compile time configuration of this srgemm kernel using OperatorClass
    using OperatorClass    = cutlass::arch::OpClassSimt;
    using SmArch           = cutlass::arch::Sm80;
  
    using AdditionOp       = cuasr::plus<float>;
    using MultiplicationOp = minussquare<float>;
    using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, float, 1>;
  
    static int constexpr AlignmentA = 1;
    static int constexpr AlignmentB = 1;
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor    = cutlass::layout::RowMajor;
    using ThreadblockShape          = cutlass::gemm::GemmShape<64, 128, 8>;
    using WarpShape                 = cutlass::gemm::GemmShape<16, 64, 8>;
    using InstructionShape          = cutlass::gemm::GemmShape<1, 1, 1>;
    using ThreadblockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    static int constexpr Stages = 2;
  

    using cuASR_SGEMM = cuasr::gemm::device::Srgemm<
        AdditionOp,         // Thread level SemiRing operator
        MultiplicationOp,   // Thread level SemiRing operator
        float,                // element type of A
        ColumnMajor,           // layout of A
        float,                // element type of B
        ColumnMajor,           // layout of B
        float,                // element t  ype of C
        ColumnMajor,           // layout of C
        float,                // element type of D
        OperatorClass,      // Logical operator class (SIMT/Tensor)
        SmArch,             // CUDA architecture
        ThreadblockShape,   // GEMM shape at CTA level
        WarpShape,          // GEMM shape at Warp level
        InstructionShape,   // GEMM shape at thread level
        EpilogueOutputOp,   // Epilogue operator at thread level
        ThreadblockSwizzle, // GEMM threadblock swizzler
        Stages,             // Pipeline stages for shmem
        AlignmentA,         // Alignment of A elements
        AlignmentB,         // Alignment of B elements
        false               // SplitKSerial
        >;
  
    float alpha = MultiplicationOp::Identity;
    float beta = do_epilogue_min ? MultiplicationOp::Annihilator : MultiplicationOp::Identity;
  
    // construct kernel arguments struct
    cuASR_SGEMM::Arguments args(
        { M, N, K },    // Problem dimensions
        { A, lda },     // Tensor-ref for source matrix A
        { B, ldb },     // Tensor-ref for source matrix B
        { C, ldc },     // Tensor-ref for source matrix C
        { D, ldc },     // Tensor-ref for destination matrix D
        { alpha, beta } //
    );
  
    // launch SRGEMM kernel
    cuASR_SGEMM srgemm;
    cutlass::Status status = srgemm(args, nullptr, stream);
    return static_cast<int>(status);
    }
