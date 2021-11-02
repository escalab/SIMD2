# NVCC_FLAGS = -I$(CUDA_DIR)/include -gencode arch=compute_61,code=compute_61 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_53,code=compute_53
CUDA_DIR = /usr/local/cuda-11.0
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -O3 -I$(CUDA_DIR)/include -I../../common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80
CUASR = ../../../cuASR

INCLUDE=-I$(CUASR)/cutlass/include/ \
				-I$(CUASR)/include/ \
				-I$(CUASR)/tools/include/ \
				-I$(CUASR)/cutlass/tools/util/include/cutlass/util/

.PHONY: all aplp_emulation

all: aplp_emulation

aplp_emulation: aplp_emulation.cu
	$(NVCC) $(NVCC_FLAGS) -o aplp_emulation aplp_emulation.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets $(INCLUDE)
clean:
	rm aplp_emulation
