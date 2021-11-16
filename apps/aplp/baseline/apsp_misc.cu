/*
 * =======================================================================
 *  This file is part of APSP-CUDA.
 *  Copyright (C) 2016 Marios Mitalidis
 *
 *  APSP-CUDA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  APSP-CUDA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with APSP-CUDA.  If not, see <http://www.gnu.org/licenses/>.
 * =======================================================================
 */ 


#include "apsp_misc.h"

const int errorMemAllocate = 101;

//free matrix data
void  matrix_free(float** mat, int N) {
	for (int i = 0; i < N; i++)
		free(mat[i]);
	free(mat);
}

// safe data allocation
void* safe_malloc(size_t size) {
	void* ptr = malloc(size);
	if (ptr == NULL) {
		fprintf(stderr,"Cannot allocate memory\n");
		exit(errorMemAllocate);
	}

	return ptr;
}

// allocate memory for matrix
float** matrix_malloc(int N) {
	float** mat = (float**) safe_malloc(N * sizeof(float*));
	for (int i = 0; i < N; ++i)
		mat[i] = (float*) safe_malloc(N * sizeof(float)); 
	return mat;
}

// check if a number is a power of 2
int isPowerOfTwo (int x) {
 return (
   x == 1 || x == 2 || x == 4 || x == 8 || x == 16 || x == 32 ||
   x == 64 || x == 128 || x == 256 || x == 512 || x == 1024 ||
   x == 2048 || x == 4096 || x == 8192 || x == 16384 ||
   x == 32768 || x == 65536 || x == 131072 || x == 262144 ||
   x == 524288 || x == 1048576 || x == 2097152 ||
   x == 4194304 || x == 8388608 || x == 16777216 ||
   x == 33554432 || x == 67108864 || x == 134217728 ||
   x == 268435456 || x == 536870912 || x == 1073741824 ||
   x == 2147483648);
}

int getPowerofTwo(int N) {
	int p = 0;
	while (N >>= 1)
		++p; 
	return p;
}
