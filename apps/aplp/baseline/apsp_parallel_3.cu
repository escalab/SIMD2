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


#include "apsp_parallel_3.h"
#include "apsp_misc.h"
#include "cuda_error_check.h"
#include "cuda.h"

#include <time.h>
#include <sys/time.h>


#define MIN_MACRO(a,b) ( (a) < (b) ? (a) : (b) )

#define NUM_ITR 5

/*
 * Max matrix size is N = 2^12 = 4096. 
 * Total number of threads that will be executed is 4096^2.
 * One for each cell.
 */

// programmer defined
const int block_dim = 64;

const int thread_dim = block_dim / 2;
const dim3 threads(thread_dim,thread_dim);

// kernel for pass 1
__global__ void apsp_parallel_3_kernel_1(float* dev_dist, int N, int stage) {

	// get indices for the multiple cells
	int i0 = stage*block_dim + threadIdx.y;
	int j0 = stage*block_dim + threadIdx.x;
	int tid0 = i0*N + j0;

	int i1 = stage*block_dim + threadIdx.y;
	int j1 = stage*block_dim + threadIdx.x + thread_dim;
	int tid1 = i1*N + j1;

	int i2 = stage*block_dim + threadIdx.y + thread_dim;
	int j2 = stage*block_dim + threadIdx.x;
	int tid2 = i2*N + j2;

	int i3 = stage*block_dim + threadIdx.y + thread_dim;
	int j3 = stage*block_dim + threadIdx.x + thread_dim;
	int tid3 = i3*N + j3;

	// allocate shared memory
	__shared__ float sd[block_dim][block_dim];

	// copy data from main memory to shared memory
	sd[threadIdx.y           ][threadIdx.x           ] = dev_dist[tid0];
	sd[threadIdx.y           ][threadIdx.x+thread_dim] = dev_dist[tid1];
	sd[threadIdx.y+thread_dim][threadIdx.x           ] = dev_dist[tid2];
	sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim] = dev_dist[tid3];
	__syncthreads();

	// iterate for the values of k
	for (int k = 0; k < block_dim; k++) {

		float vertex0   = sd[threadIdx.y][threadIdx.x];
		float alt_path0 = sd[k][threadIdx.x] + sd[threadIdx.y][k];

		float vertex1   = sd[threadIdx.y][threadIdx.x+thread_dim];
		float alt_path1 = sd[k][threadIdx.x+thread_dim] + sd[threadIdx.y][k];

		float vertex2   = sd[threadIdx.y+thread_dim][threadIdx.x]; float alt_path2 = sd[k][threadIdx.x] + sd[threadIdx.y+thread_dim][k]; 
		float vertex3   = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
		float alt_path3 = sd[k][threadIdx.x+thread_dim] + sd[threadIdx.y+thread_dim][k];

		sd[threadIdx.y][threadIdx.x]            = MIN_MACRO( vertex0, alt_path0 );
		sd[threadIdx.y][threadIdx.x+thread_dim] = MIN_MACRO( vertex1, alt_path1 );
		sd[threadIdx.y+thread_dim][threadIdx.x] = MIN_MACRO( vertex2, alt_path2 );
		sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim] = 
                                                         MIN_MACRO( vertex3, alt_path3 );
		__syncthreads();

	}

	// write result back to main memory
	dev_dist[tid0] = sd[threadIdx.y           ][threadIdx.x           ];
	dev_dist[tid1] = sd[threadIdx.y           ][threadIdx.x+thread_dim];
	dev_dist[tid2] = sd[threadIdx.y+thread_dim][threadIdx.x           ];
	dev_dist[tid3] = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
}

// kernel for pass 2
__global__ void apsp_parallel_3_kernel_2(float* dev_dist, int N, int stage) {
	
	// get indices of the current block
	int skip_center_block = MIN_MACRO( (blockIdx.x+1)/(stage+1), 1 );

	int box_y = 0;
	int box_x = 0;

	// block in the same row with the primary block
	if (blockIdx.y == 0) {
		box_y = stage;
		box_x = blockIdx.x + skip_center_block;
	
	}
	// block in the same column with the primary block
	else {
		box_y = blockIdx.x + skip_center_block;
		box_x = stage;
	}

	// get indices for the current cell
	int i0 = box_y * block_dim + threadIdx.y;
	int j0 = box_x * block_dim + threadIdx.x;

	int i1 = box_y * block_dim + threadIdx.y;
	int j1 = box_x * block_dim + threadIdx.x + thread_dim;

	int i2 = box_y * block_dim + threadIdx.y + thread_dim;
	int j2 = box_x * block_dim + threadIdx.x;

	int i3 = box_y * block_dim + threadIdx.y + thread_dim;
	int j3 = box_x * block_dim + threadIdx.x + thread_dim;

	// get indices for the cell of the primary block
	int pi0 = stage*block_dim + threadIdx.y;
	int pj0 = stage*block_dim + threadIdx.x;

	int pi1 = stage*block_dim + threadIdx.y;
	int pj1 = stage*block_dim + threadIdx.x + thread_dim;

	int pi2 = stage*block_dim + threadIdx.y + thread_dim;
	int pj2 = stage*block_dim + threadIdx.x;

	int pi3 = stage*block_dim + threadIdx.y + thread_dim;
	int pj3 = stage*block_dim + threadIdx.x + thread_dim;

	// get indices of the cells from the device main memory
	int tid0 = i0*N + j0;
	int ptid0 = pi0*N + pj0;

	int tid1 = i1*N + j1;
	int ptid1 = pi1*N + pj1;

	int tid2 = i2*N + j2;
	int ptid2 = pi2*N + pj2;

	int tid3 = i3*N + j3;
	int ptid3 = pi3*N + pj3;

	// allocate shared memory
	__shared__ float sd[block_dim][2*block_dim];

	// copy current block and primary block to shared memory
	sd[threadIdx.y][threadIdx.x]             = dev_dist[tid0];
	sd[threadIdx.y][block_dim + threadIdx.x] = dev_dist[ptid0]; 

	sd[threadIdx.y][threadIdx.x + thread_dim]             = dev_dist[tid1];
	sd[threadIdx.y][block_dim + threadIdx.x + thread_dim] = dev_dist[ptid1]; 

	sd[threadIdx.y+thread_dim][threadIdx.x]             = dev_dist[tid2];
	sd[threadIdx.y+thread_dim][block_dim + threadIdx.x] = dev_dist[ptid2]; 

	sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim]             = dev_dist[tid3];
	sd[threadIdx.y+thread_dim][block_dim + threadIdx.x+thread_dim] = dev_dist[ptid3]; 

	__syncthreads();

	// block in the same row with the primary block
	if (blockIdx.y == 0) {
		for (int k = 0; k < block_dim; k++) {

			float vertex0   = sd[threadIdx.y][threadIdx.x];
			float alt_path0 = sd[k][threadIdx.x] 
                               	              + sd[threadIdx.y][block_dim + k];

			float vertex1   = sd[threadIdx.y][threadIdx.x+thread_dim];
			float alt_path1 = sd[k][threadIdx.x+thread_dim] 
                               	              + sd[threadIdx.y][block_dim + k];

			float vertex2   = sd[threadIdx.y+thread_dim][threadIdx.x];
			float alt_path2 = sd[k][threadIdx.x] 
                               	              + sd[threadIdx.y+thread_dim][block_dim + k];

			float vertex3   = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
			float alt_path3 = sd[k][threadIdx.x+thread_dim] 
                               	              + sd[threadIdx.y+thread_dim][block_dim + k];

			sd[threadIdx.y][threadIdx.x]            = MIN_MACRO( vertex0, alt_path0 );
			sd[threadIdx.y][threadIdx.x+thread_dim] = MIN_MACRO( vertex1, alt_path1 );
			sd[threadIdx.y+thread_dim][threadIdx.x] = MIN_MACRO( vertex2, alt_path2 );
			sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim] 
                                                                = MIN_MACRO( vertex3, alt_path3 );
			__syncthreads();

		}
	}
	// block in the same column with the primary block
	else {
		for (int k = 0; k < block_dim; k++) {

			float vertex0   = sd[threadIdx.y][threadIdx.x];
			float alt_path0 = sd[threadIdx.y][k] 
                                           + sd[k][block_dim + threadIdx.x];

			float vertex1   = sd[threadIdx.y][threadIdx.x+thread_dim];
			float alt_path1 = sd[threadIdx.y][k] 
                                           + sd[k][block_dim + threadIdx.x+thread_dim];

			float vertex2   = sd[threadIdx.y+thread_dim][threadIdx.x];
			float alt_path2 = sd[threadIdx.y+thread_dim][k] 
                                           + sd[k][block_dim + threadIdx.x];

			float vertex3   = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
			float alt_path3 = sd[threadIdx.y+thread_dim][k] 
                                           + sd[k][block_dim + threadIdx.x+thread_dim];

			sd[threadIdx.y][threadIdx.x]            = MIN_MACRO( vertex0, alt_path0 );
			sd[threadIdx.y][threadIdx.x+thread_dim] = MIN_MACRO( vertex1, alt_path1 );
			sd[threadIdx.y+thread_dim][threadIdx.x] = MIN_MACRO( vertex2, alt_path2 );
			sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim] 
                                                                = MIN_MACRO( vertex3, alt_path3 );
			__syncthreads();
		}
	}

	// write result back to main memory
	dev_dist[tid0] = sd[threadIdx.y           ][threadIdx.x           ];
	dev_dist[tid1] = sd[threadIdx.y           ][threadIdx.x+thread_dim];
	dev_dist[tid2] = sd[threadIdx.y+thread_dim][threadIdx.x           ];
	dev_dist[tid3] = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
}

// kernel for pass 3
__global__ void apsp_parallel_3_kernel_3(float* dev_dist, int N, int stage) {

	// get indices of the current block
	int skip_center_block_y = MIN_MACRO( (blockIdx.y+1)/(stage+1), 1 );
	int skip_center_block_x = MIN_MACRO( (blockIdx.x+1)/(stage+1), 1 );

	int box_y = blockIdx.y + skip_center_block_y;
	int box_x = blockIdx.x + skip_center_block_x;

	// get indices for the multiple cells
	int i0 = box_y * block_dim + threadIdx.y;
	int j0 = box_x * block_dim + threadIdx.x;

	int i1 = box_y * block_dim + threadIdx.y;
	int j1 = box_x * block_dim + threadIdx.x + thread_dim;

	int i2 = box_y * block_dim + threadIdx.y + thread_dim;
	int j2 = box_x * block_dim + threadIdx.x;

	int i3 = box_y * block_dim + threadIdx.y + thread_dim;
	int j3 = box_x * block_dim + threadIdx.x + thread_dim;

	// get indices from the cell in the same row with the current box
	int ri0 = i0;
	int rj0 = stage*block_dim + threadIdx.x;
	
	int ri1 = i1;
	int rj1 = stage*block_dim + threadIdx.x + thread_dim;
	
	int ri2 = i2;
	int rj2 = stage*block_dim + threadIdx.x;
	
	int ri3 = i3;
	int rj3 = stage*block_dim + threadIdx.x + thread_dim;
	
	// get indices from the cell in the same column with the current box
	int ci0 = stage*block_dim + threadIdx.y;
	int cj0 = j0;

	int ci1 = stage*block_dim + threadIdx.y;
	int cj1 = j1;

	int ci2 = stage*block_dim + threadIdx.y + thread_dim;
	int cj2 = j2;

	int ci3 = stage*block_dim + threadIdx.y + thread_dim;
	int cj3 = j3;

	// get indices of the cells from the device main memory
	int  tid0 =  i0*N +  j0;
	int rtid0 = ri0*N + rj0;
	int ctid0 = ci0*N + cj0;

	int  tid1 =  i1*N +  j1;
	int rtid1 = ri1*N + rj1;
	int ctid1 = ci1*N + cj1;

	int  tid2 =  i2*N +  j2;
	int rtid2 = ri2*N + rj2;
	int ctid2 = ci2*N + cj2;

	int  tid3 =  i3*N +  j3;
	int rtid3 = ri3*N + rj3;
	int ctid3 = ci3*N + cj3;

	// allocate shared memory
	__shared__ float sd[block_dim][3*block_dim];

	// copy current block and depending blocks to shared memory
	sd[threadIdx.y][threadIdx.x              ] = dev_dist[tid0];
	sd[threadIdx.y][  block_dim + threadIdx.x] = dev_dist[rtid0]; 
	sd[threadIdx.y][2*block_dim + threadIdx.x] = dev_dist[ctid0]; 

	sd[threadIdx.y][threadIdx.x + thread_dim              ] = dev_dist[tid1];
	sd[threadIdx.y][  block_dim + threadIdx.x + thread_dim] = dev_dist[rtid1]; 
	sd[threadIdx.y][2*block_dim + threadIdx.x + thread_dim] = dev_dist[ctid1]; 

	sd[threadIdx.y + thread_dim][threadIdx.x              ] = dev_dist[tid2];
	sd[threadIdx.y + thread_dim][  block_dim + threadIdx.x] = dev_dist[rtid2]; 
	sd[threadIdx.y + thread_dim][2*block_dim + threadIdx.x] = dev_dist[ctid2]; 

	sd[threadIdx.y + thread_dim][threadIdx.x + thread_dim              ] = dev_dist[tid3];
	sd[threadIdx.y + thread_dim][  block_dim + threadIdx.x + thread_dim] = dev_dist[rtid3]; 
	sd[threadIdx.y + thread_dim][2*block_dim + threadIdx.x + thread_dim] = dev_dist[ctid3]; 
	__syncthreads();

	for (int k = 0; k < block_dim; k++) {

		float vertex0   = sd[threadIdx.y][threadIdx.x];
		float alt_path0 = sd[threadIdx.y][block_dim + k]
       	        	             + sd[k][2*block_dim + threadIdx.x];

		float vertex1   = sd[threadIdx.y][threadIdx.x+thread_dim];
		float alt_path1 = sd[threadIdx.y][block_dim + k]
       	        	             + sd[k][2*block_dim + threadIdx.x+thread_dim];

		float vertex2   = sd[threadIdx.y+thread_dim][threadIdx.x];
		float alt_path2 = sd[threadIdx.y+thread_dim][block_dim + k]
       	        	             + sd[k][2*block_dim + threadIdx.x];

		float vertex3   = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
		float alt_path3 = sd[threadIdx.y+thread_dim][block_dim + k]
       	        	             + sd[k][2*block_dim + threadIdx.x+thread_dim];

		sd[threadIdx.y][threadIdx.x]            = MIN_MACRO( vertex0, alt_path0 );
		sd[threadIdx.y][threadIdx.x+thread_dim] = MIN_MACRO( vertex1, alt_path1 );
		sd[threadIdx.y+thread_dim][threadIdx.x] = MIN_MACRO( vertex2, alt_path2 );
		sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim] 
                                                        = MIN_MACRO( vertex3, alt_path3 );
		__syncthreads();
	}

	// write result back to main memory
	dev_dist[tid0] = sd[threadIdx.y           ][threadIdx.x           ];
	dev_dist[tid1] = sd[threadIdx.y           ][threadIdx.x+thread_dim];
	dev_dist[tid2] = sd[threadIdx.y+thread_dim][threadIdx.x           ];
	dev_dist[tid3] = sd[threadIdx.y+thread_dim][threadIdx.x+thread_dim];
}

/*
 * Solves the all-pairs shortest path problem using Floyd Warshall algorithm
 *
 * Implementation based on:
 * All-Pairs Shortest-Paths for Large Graphs on the GPU
 * Gary J. Katz and Joseph T. Kider Jr.
*/
unsigned int apsp_parallel_3(float** graph, float** dist, int N) {
  // timer
  struct timeval startwtime, endwtime;
  unsigned int tot_time = 0;
  for (int itr = 0; itr < NUM_ITR; itr++){
      // check the dimension of the input matrix
    if (!isPowerOfTwo(N) || N < 128) {
      return (apsp_parallel_3_status::invalid_dimension);
    }

    // allocate memory on the device
    float* dev_dist;
    gpuErrchk( cudaMalloc( (void**)&dev_dist, N*N * sizeof (float) ) );

    // initialize dist matrix on device
    for (int i = 0; i < N; i++)
      gpuErrchk( cudaMemcpy(dev_dist +i*N, graph[i], N * sizeof (float),
                  cudaMemcpyHostToDevice) );
    // get the power of 2 of the dimension
    int p = getPowerofTwo(N);
    int r = getPowerofTwo(block_dim);

    int nBlocks = 1 << (p-r);
    
    // get the dimensions of the grid
    dim3 blocks1(1);
    dim3 blocks2(nBlocks-1,2);
    dim3 blocks3(nBlocks-1, nBlocks-1);

    // start measuring time
    gettimeofday(&startwtime,NULL);
    // For each element of the vertex set
    for (int stage = 0; stage < nBlocks; stage++) {

      // pass 1 - launch kernel 1
      apsp_parallel_3_kernel_1<<<blocks1,threads>>>(dev_dist,N,stage);
      gpuKerchk();

      // pass 2 - launch kernel 2
      apsp_parallel_3_kernel_2<<<blocks2,threads>>>(dev_dist,N,stage);
      gpuKerchk();

      // pass 3 - launch kernel 3
      apsp_parallel_3_kernel_3<<<blocks3,threads>>>(dev_dist,N,stage);
      gpuKerchk();
    }
    cudaDeviceSynchronize();

    // stop measuring time
	  gettimeofday(&endwtime,NULL);
    tot_time += (endwtime.tv_sec * 1000000 + endwtime.tv_usec) - \
                                        (startwtime.tv_sec * 1000000 + startwtime.tv_usec);

    // return results to dist matrix on host
    for (int i = 0; i < N; i++)
      gpuErrchk( cudaMemcpy(dist[i], dev_dist +i*N, N * sizeof (float),
                  cudaMemcpyDeviceToHost) );
    gpuErrchk(cudaFree(dev_dist));
  }
  return tot_time;
	// return (apsp_parallel_3_status::success);
}
 
