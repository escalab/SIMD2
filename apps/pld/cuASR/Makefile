include ../../../config.mk
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I../../../utils/common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80

CUASR = ../../../cuASR
INCLUDE=-I$(CUASR)/cutlass/include/ \
				-I$(CUASR)/include/ \
				-I$(CUASR)/tools/include/ \
				-I$(CUASR)/cutlass/tools/util/include/cutlass/util/

.PHONY: all test

all: knn_cuASR

knn_cuASR: test.cpp knncuda.cu
	$(NVCC) $(NVCC_FLAGS) -o knn_cuASR test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets $(INCLUDE)

clean:
	rm knn_cuASR
