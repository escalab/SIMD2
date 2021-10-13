#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <math.h>

/**
  mmo interface
    1. gives host side reference for srgemm computation.
    2. A general interface that hardware needs to support for srgemm.

    ** Note **
    minussquare may not full-fill semiring properties but has the same data-flow as other srgemm kerenls.
**/
namespace mmo {
    
    template<typename T>
    void host_srgemm(
      int M,
      int N,
      int K,
      T * A,
      int lda, 
      T * B, 
      int ldb,
      T * C,
      int ldc,
      T * D,
      bool do_epilogue,  
      T (*addop)(T,T), 
      T (*mulop)(T,T))
    {
      // perform srgemm on host
      int i,j,k;
      for (i = 0; i < M; i++){
          for (j = 0; j < N; j++){
              for(k = 0; k < K; k++){
                  D[i*N+j] = addop(D[i*N+j], mulop(A[i*K+k], B[k*N+j]));
              }
          }
      }
      // accumulation if needed
      if (do_epilogue){
        for (i = 0; i < M; i++){
          for (j = 0; j < N; j++){
            D[i*N+j] = addop(D[i*N+j],C[i*N+j]);
          } 
        }
      }
    }

    // operators to support srgemm
    template<typename T>
    T plus(T lhs, T rhs){
        return lhs + rhs;
    }

    template<typename T>
    T multiplies(T lhs, T rhs){
        return lhs * rhs;
    }

    template<typename T>
    T minimum(T lhs, T rhs){
        return std::min(lhs, rhs);
    }

    template<typename T>
    T maximum(T lhs, T rhs){
        return std::max(lhs, rhs);
    }

    template<typename T>
    T minussquare(T lhs, T rhs){
        return (lhs - rhs) * (lhs - rhs);
    }

    template<typename T>
    T bin_and(T lhs, T rhs){
        return lhs && rhs;
    }

    template<typename T>
    T bin_or(T lhs, T rhs){
        return lhs || rhs;
    }

    // debug ops

    template<typename T>
    T get_rhs(T lhs, T rhs){
        return rhs;
    }

    template<typename T>
    T minus(T lhs, T rhs){
        return 15;
    }


}


