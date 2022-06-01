include ../../../config.mk
NVCC = $(CUDA_DIR)/bin/nvcc
all: ecl-apsp

ecl-apsp: ECL-APSP_10.cu ECLgraph.h
	$(NVCC) -O3 -arch=sm_80 ECL-APSP_10.cu -o ecl-apsp

clean:
	rm -f ecl-apsp
