# =======================================================================
#  This file is part of APSP-CUDA.
#  Copyright (C) 2016 Marios Mitalidis
#
#  APSP-CUDA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  APSP-CUDA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with APSP-CUDA.  If not, see <http://www.gnu.org/licenses/>.
# =======================================================================

CUDA_DIR	= /usr/local/cuda-11.1
NVCC		= $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_DIR)/include -arch=sm_80
LD_FLAGS    = -lcudart -L$(CUDA_DIR)/lib64 -lcublas

all: apsp-cuda-v3

apsp-cuda-v3: apsp-cuda-v3.o apsp_parallel_3.o apsp_misc.o cuda_error_check.o
	$(NVCC) -lcudart -o aplp_baseline cuda_error_check.o apsp_parallel_3.o apsp_misc.o apsp-cuda-v3.o

apsp-cuda-v3.o: apsp-cuda-v3.cpp
	gcc -c -fPIC -o apsp-cuda-v3.o apsp-cuda-v3.cpp

apsp_parallel_3.o: apsp_parallel_3.cu
	$(NVCC) -c -Xcompiler -fPIC apsp_parallel_3.cu -o apsp_parallel_3.o $(LD_FLAGS)

apsp_misc.o: apsp_misc.cu
	$(NVCC) -c -Xcompiler -fPIC apsp_misc.cu -o apsp_misc.o $(LD_FLAGS)

cuda_error_check.o: cuda_error_check.cu
	$(NVCC) -c -Xcompiler -fPIC cuda_error_check.cu -o cuda_error_check.o $(LD_FLAGS)

clean:
	rm -f ./*.o
	rm -f ./*.gch
	rm apsp_baseline

cleanAll:
	rm -f ./*.o 
