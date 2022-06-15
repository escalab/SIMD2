# **SIMD**^2
This repo conatins artifacts of the SIMD^2(SIMD square) project. 
1. Aplied `cuASR` as baseline kernel of srgemm (semiring gemm) computaion, to exploit potential speed-up of a tensor-core-liked hardware accelerator.
2. Studied the sparsity(density) threshold of gemm-liked computation kernel based on perfomance of `cuSparse_spgemm` and `cublas_gemmEx`.
4. Collected a set of benchmark applications that can be accelerated with SIMD^2 and shows the speed-up from emulation result.

# Enviroment
| | |
|---------|-----------|
|`OS` | Ubuntu 20.04 LTS|
|`CPU` | AMD Ryzen 3700X  |
|`GPU` | Nvidia 3080, cuda 11.1|
| `GCC`| 7.5 |


# Dependencies:
CUDA 11.x is required.

define installed cuda version in `config.mk`, e.g.:
```
CUDA_DIR = /usr/local/cuda-11.7
```

cuSparsLt: https://developer.nvidia.com/cusparselt/downloads

Python 3.8
# Build
```
git submodule update --init --recursive
make
```
# Microbenchmark
To run micro benchmark:
```
./run_bench
```

# Benchamrk Applications
To run micro benchmark:
```
./run_app
```

# Computation kernel and operators that SIMD^2 supports
|operators | Computation kernel |
|-|-|
|`min_plus` | All-Pairs Shortest Paths
|`plus_mul` | Matrix Multiplication
|`max_plus` | All-Pairs Longest Paths
|`max_min` | Maximum Capacity Problem
|`max_mul` | Maximum Reliability Paths Problem
|`min_mul` | Minimum Reliability Paths Problem
|`min_max` | Minimum Spanning Tree Problem
|`or_and` | Graph Transitive Closure
|`minus_square` | Pair-wise L2 Distance

# Software Abstraction
```cpp
for (i = 0; i < M; i++){
    for (j = 0; j < N; j++){
        for(k = 0; k < K; k++){
            C[i*N+j] = addop(C[i*N+j], mulop(A[i*K+k], B[k*N+j]));
        }
    }
}

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
```
