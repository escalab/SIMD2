# cusparseLt - `cusparseMatMul`

## Description

This sample demonstrates the usage of `cuSPARSELt` library and `cusparseMatMul` APIs for performing *structured matrix - dense matrix multiplication* by exploiting NVIDIA *Sparse Tensor Cores*, where the structured matrix is compressed with 50% sparsity ratio.

[cusparseLt Documentation](https://docs.nvidia.com/cuda/cusparselt/index.html)

<center>

`C = alpha * A * B + beta * C`

</center>

where `A`, `B`, `C` are dense matrices

## Building

* Linux
    ```bash
    make
    ```

* or in alternative:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## Support

* **Supported SM Architectures:** SM 8.0, SM 8.6
* **Supported OSes:** Linux, Windows
* **Supported CPU Architectures**: x86_64, arm64
* **Supported Compilers**: gcc, clang, Intel icc, IBM xlc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C++14`

## Prerequisites

* [CUDA 11.2 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [cusparseLt](https://developer.nvidia.com/cusparselt/downloads)
* [CMake 3.9](https://cmake.org/download/) or above
