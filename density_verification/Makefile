CUDA_DIR = /usr/local/cuda-11.1
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I../utils/common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80

.PHONY: all spdense

all: spdense

spdense:
	$(NVCC) $(NVCC_FLAGS) -o spdense sp_gemm_comp.cu -lcuda -lcublas -lcusparse -lcudart -Wno-deprecated-gpu-targets

clean:
	rm test
