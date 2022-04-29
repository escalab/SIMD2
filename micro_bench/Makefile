CUDA_DIR	= /usr/local/cuda-11.1
NVCC		= $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_DIR)/include -arch=sm_80
LD_FLAGS    = -lcudart -L$(CUDA_DIR)/lib64 -lcublas -lcublasLt

INCLUDE=-I../cuASR/cutlass/include/ \
				-I../cuASR/include/ \
				-I../cuASR/tools/include/ \
				-I../cuASR/cutlass/tools/util/include/cutlass/util/

COMMON=-I../utils/common/

				
default: all

all: srgemm_bench

srgemm_bench:	srgemm_bench.cu ../kernel/srgemm.cuh ../kernel/tensor_srgemm.cuh
	$(NVCC) -O3 -o srgemm_bench srgemm_bench.cu $(INCLUDE) $(NVCC_FLAGS) $(LD_FLAGS)

clean:
	rm srgemm_bench