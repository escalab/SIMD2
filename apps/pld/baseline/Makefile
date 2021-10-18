# NVCC_FLAGS = -I$(CUDA_DIR)/include -gencode arch=compute_61,code=compute_61 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_53,code=compute_53
CUDA_DIR = /usr/local/cuda-11.0
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I../../../utils/common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80

.PHONY: all test

all: knn_baseline

knn_baseline: test.cpp knncuda.cu
	$(NVCC) $(NVCC_FLAGS) -o knn_baseline test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

clean:
	rm test
