include ../../../config.mk
NVCC = $(CUDA_DIR)/bin/nvcc
all: ecl-aplp

ecl-aplp: ECL-APSP_10.cu ECLgraph.h
	$(NVCC) -O3 -arch=sm_80 ECL-APSP_10.cu -o ecl-aplp

clean:
	rm -f ecl-aplp
