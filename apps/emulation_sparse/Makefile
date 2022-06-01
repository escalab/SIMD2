include ../../config.mk
NVCC_FLAGS = -I$(CUDA_DIR)/include -gencode arch=compute_61,code=compute_61 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_53,code=compute_53
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -O3 --std=c++14
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80
CUASR = ../../../cuASR
CUSPARSELT_PATH = /nfshome/yuz057/cuSparseLt/libcusparse_lt

INCLUDE=-I../../common/inc

CUASR_INCLUDE=-I$(CUASR)/cutlass/include/ \
				-I$(CUASR)/include/ \
				-I$(CUASR)/tools/include/ \
				-I$(CUASR)/cutlass/tools/util/include/cutlass/util/ \
				-I${CUSPARSELT_PATH}/include

CUSPLT_INCLUDE=-I$(CUDA_DIR)/include -I${CUSPARSELT_PATH}/include

NVRTC_SHARED := ${CUDA_DIR}/targets/x86_64-linux/lib/libnvrtc.so

ifeq ($(shell uname -m), aarch64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "sbsa-linux"
endif
endif
ifeq ($(shell uname -m), x86_64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "x86_64-linux"
endif
endif
NVRTC_SHARED := ${CUDA_DIR}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so
INCS         := -I$(CUDA_DIR)/include -I${CUSPARSELT_PATH}/include
LIBS         := -lcuda -lcudart -lcusparse -lcublas -ldl ${NVRTC_SHARED}

.PHONY: all emulation_sparse

all: emulation_sparse

# apsp_emulation_sparse: apsp_emulation_sparse.cu
# 	$(NVCC)  $(NVCC_FLAGS)  \
# 	-o apsp_emulation_sparse apsp_emulation_sparse.cu \
# 	-lcuda  -lcudart -lcusparse -lcublas -lcusparseLt -ldl ${NVRTC_SHARED} \
# 	-I$(CUDA_DIR)/include $(CUASR_INCLUDE) $(CUSPLT_INCLUDE) $(INCLUDE) \
# 	-L${CUSPARSELT_PATH}/lib64  ${LIBS}

emulation_sparse: emulation_sparse.cpp
	$(NVCC)  ${INCS}  emulation_sparse.cpp -o emulation_sparse  \
         -L${CUSPARSELT_PATH}/lib64  -lcusparseLt ${LIBS} -gencode arch=compute_80,code=compute_80

clean:
	rm emulation_sparse
