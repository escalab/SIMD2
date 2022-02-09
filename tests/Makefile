CUDA_DIR	= /usr/local/cuda-11.1
NVCC		= $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_DIR)/include -arch=sm_80
LD_FLAGS    = -lcudart -L$(CUDA_DIR)/lib64 -lcublas

INCLUDE=-I../cuASR/cutlass/include/ \
				-I../cuASR/include/ \
				-I../cuASR/tools/include/ \
				-I../cuASR/cutlass/tools/util/include/cutlass/util/

COMMON=-I../utils/common/

				
default: all

all: cuAsr_kernel_test host_srgemm_test

cuAsr_kernel_test:	cuAsr_kernel_test.cu ../kernel/host_srgemm.h ../kernel/srgemm.cuh
	$(NVCC) -o cuAsr_kernel_test cuAsr_kernel_test.cu $(INCLUDE)

host_srgemm_test: host_srgemm_test.cpp ../kernel/host_srgemm.h
	g++  -o host_srgemm_test host_srgemm_test.cpp

tensor_srgemm_test:	tensor_srgemm_test.cu ../kernel/host_srgemm.h ../kernel/tensor_srgemm.cuh
	$(NVCC) -o  tensor_srgemm_test tensor_srgemm_test.cu $(NVCC_FLAGS) $(LD_FLAGS)

mem_test: mem_sens_test.cu
	$(NVCC) -o mem_sens_test mem_sens_test.cu
clean:
	rm cuAsr_kernel_test
	rm host_srgemm_test
	rm tensor_srgemm_test