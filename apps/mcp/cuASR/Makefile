include ../../../config.mk
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I../../common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80
CUASR = ../../../cuASR

INCLUDE=-I$(CUASR)/cutlass/include/ \
				-I$(CUASR)/include/ \
				-I$(CUASR)/tools/include/ \
				-I$(CUASR)/cutlass/tools/util/include/cutlass/util/

.PHONY: all mcp_cuASR

all: mcp_cuASR

mcp_cuASR: mcp_cuASR.cu
	$(NVCC) $(NVCC_FLAGS) -o mcp_cuASR mcp_cuASR.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets $(INCLUDE)
clean:
	rm mcp_cuASR
