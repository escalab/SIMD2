include ../../../config.mk
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -O3 -I$(CUDA_DIR)/include -I../../common/inc
NVCC_FLAGS += -gencode arch=compute_80,code=compute_80
CUASR = ../../../cuASR

INCLUDE=-I$(CUASR)/cutlass/include/ \
				-I$(CUASR)/include/ \
				-I$(CUASR)/tools/include/ \
				-I$(CUASR)/cutlass/tools/util/include/cutlass/util/

.PHONY: all minrp_emulation

all: minrp_emulation

minrp_emulation: minrp_emulation.cu
	$(NVCC) $(NVCC_FLAGS) -o minrp_emulation minrp_emulation.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets $(INCLUDE)
clean:
	rm minrp_emulation
