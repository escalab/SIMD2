include ../../../config.mk
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I../../../utils/common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80

.PHONY: all test

all: knn_baseline

knn_baseline: test.cpp knncuda.cu
	$(NVCC) $(NVCC_FLAGS) -o knn_baseline test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

clean:
	rm knn_baseline
