/*
ECL-APSP code: ECL-APSP is an all-pairs-shortest-paths CUDA implementation of
the Floyd-Warshall algorithm that is quite fast. It operates on graphs stored
in binary CSR format.

Copyright (c) 2021, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Yiqian Liu and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/ECL-APSP/.
*/


#include <cstdio>
#include <limits>
#include <sys/time.h>
#include <cuda.h>
#include "ECLgraph.h"

using mtype = int;

static const int ws = 32; // warp size
static const int tile = 64; // tile size
static const int ThreadsPerBlock = ws * ws;
static const int dev = 0; // device

// initialize adj matrix
static __global__ void init1(const int nodes, mtype* const __restrict__ AdjMat, const int upper)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int i = idx / upper;
  if (i < upper) {
    const int j = idx % upper;
    AdjMat[idx] = ((i == j) && (i < nodes)) ? 0 : (INT_MAX / 2);
  }
}

// add edges to adj matrix
static __global__ void init2(const ECLgraph g, mtype* const __restrict__ AdjMat, const int upper)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < g.nodes) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      AdjMat[i * upper + nei] = g.eweight[j];
    }
  }
}

// calculate the first tile
static __global__ __launch_bounds__(ThreadsPerBlock, 1)
void FW0_64(mtype* const __restrict__ AdjMat, const int upper, mtype* const __restrict__ krows, mtype* const __restrict__ kcols)
{
  __shared__ mtype temp[tile * tile];
  __shared__ mtype krow[tile * tile];

  const int warp_a = threadIdx.x / ws; // i: 0-31, upper half
  const int warp_b = warp_a + ws; // i: 32-63, lower half
  const int lane_a = threadIdx.x % ws; // j: 0-31, left half
  const int lane_b = lane_a + ws; // j: 32-63, right half

  const int idx0_aa = warp_a * upper + lane_a; // upper left
  const int idx0_ab = warp_a * upper + lane_b; // upper right
  const int idx0_ba = warp_b * upper + lane_a; // lower left
  const int idx0_bb = warp_b * upper + lane_b; // lower right

  const int idx1_aa = lane_a * tile + warp_a;
  const int idx1_ab = lane_b * tile + warp_a;
  const int idx1_ba = lane_a * tile + warp_b;
  const int idx1_bb = lane_b * tile + warp_b;

  int idx2_a = lane_a;
  int idx2_b = lane_b;

  mtype ij_aa = AdjMat[idx0_aa];
  mtype ij_ab = AdjMat[idx0_ab];
  mtype ij_ba = AdjMat[idx0_ba];
  mtype ij_bb = AdjMat[idx0_bb];

  # pragma unroll 64
  for (int k = 0; k < tile; k++) {
    if (warp_a == k) krow[idx2_a] = ij_aa;
    if (warp_a == k) krow[idx2_b] = ij_ab;
    if (warp_b == k) krow[idx2_a] = ij_ba;
    if (warp_b == k) krow[idx2_b] = ij_bb;
    __syncthreads();

    mtype ik_a, ik_b;
    if (k < ws) {
      ik_a = __shfl_sync(~0, ij_aa, k);
      ik_b = __shfl_sync(~0, ij_ba, k);
    } else {
      ik_a = __shfl_sync(~0, ij_ab, k - ws);
      ik_b = __shfl_sync(~0, ij_bb, k - ws);
    }

    const mtype kr_a = krow[idx2_a];
    const mtype kr_b = krow[idx2_b];

    ij_aa = min(ij_aa, ik_a + kr_a);
    ij_ab = min(ij_ab, ik_a + kr_b);
    ij_ba = min(ij_ba, ik_b + kr_a);
    ij_bb = min(ij_bb, ik_b + kr_b);

    if (warp_a == k) krows[idx0_aa] = ij_aa;
    if (warp_a == k) krows[idx0_ab] = ij_ab;
    if (warp_b == k) krows[idx0_ba] = ij_ba;
    if (warp_b == k) krows[idx0_bb] = ij_bb;

    if (lane_a == k) temp[idx1_aa] = ij_aa;
    if (lane_a == k) temp[idx1_ba] = ij_ba;
    if (lane_b == k) temp[idx1_ab] = ij_ab;
    if (lane_b == k) temp[idx1_bb] = ij_bb;

    idx2_a += tile;
    idx2_b += tile;
  }

  __syncthreads();
  kcols[idx0_aa] = temp[warp_a * tile + lane_a];
  kcols[idx0_ab] = temp[warp_a * tile + lane_b];
  kcols[idx0_ba] = temp[warp_b * tile + lane_a];
  kcols[idx0_bb] = temp[warp_b * tile + lane_b];
  AdjMat[idx0_aa] = ij_aa;
  AdjMat[idx0_ab] = ij_ab;
  AdjMat[idx0_ba] = ij_ba;
  AdjMat[idx0_bb] = ij_bb;
}

// calculate k^th row and col
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void FWrowcol_64(mtype* const __restrict__ AdjMat, const int upper, mtype* const __restrict__ krows, mtype* const __restrict__ kcols, const int x, const int subm1)
{
  __shared__ mtype temp[tile * tile];
  __shared__ mtype krow[tile * tile];

  const int warp_a = threadIdx.x / ws; // i: 0-31, upper half
  const int warp_b = warp_a + ws; // i: 32-63, lower half
  const int lane_a = threadIdx.x % ws; // j: 0-31, left half
  const int lane_b = lane_a + ws; // j: 32-63, right half

  int y = blockIdx.x;

  if (y < subm1) {
    if (y >= x) y++;
    const int i_a = warp_a + x * tile;
    const int i_b = warp_b + x * tile;
    const int j_a = lane_a + y * tile;
    const int j_b = lane_b + y * tile;

    const int idx0_aa = i_a * upper + j_a;
    const int idx0_ab = i_a * upper + j_b;
    const int idx0_ba = i_b * upper + j_a;
    const int idx0_bb = i_b * upper + j_b;

    int idx1_a = warp_a;
    int idx1_b = warp_b;
    int idx2_a = lane_a;
    int idx2_b = lane_b;

    temp[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile];
    temp[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile];
    temp[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile];
    temp[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile];
    __syncthreads();

    mtype ij_aa = AdjMat[idx0_aa];
    mtype ij_ab = AdjMat[idx0_ab];
    mtype ij_ba = AdjMat[idx0_ba];
    mtype ij_bb = AdjMat[idx0_bb];

    const mtype orig_aa = ij_aa;
    const mtype orig_ab = ij_ab;
    const mtype orig_ba = ij_ba;
    const mtype orig_bb = ij_bb;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      if (warp_a == k) krow[idx2_a] = ij_aa;
      if (warp_a == k) krow[idx2_b] = ij_ab;
      if (warp_b == k) krow[idx2_a] = ij_ba;
      if (warp_b == k) krow[idx2_b] = ij_bb;
      __syncthreads();

      const mtype ik_a = temp[idx1_a];
      const mtype ik_b = temp[idx1_b];
      const mtype kr_a = krow[idx2_a];
      const mtype kr_b = krow[idx2_b];

      ij_aa = min(ij_aa, ik_a + kr_a);
      ij_ab = min(ij_ab, ik_a + kr_b);
      ij_ba = min(ij_ba, ik_b + kr_a);
      ij_bb = min(ij_bb, ik_b + kr_b);

      if (warp_a == k) krows[idx0_aa] = ij_aa;
      if (warp_a == k) krows[idx0_ab] = ij_ab;
      if (warp_b == k) krows[idx0_ba] = ij_ba;
      if (warp_b == k) krows[idx0_bb] = ij_bb;

      idx1_a += tile;
      idx1_b += tile;
      idx2_a += tile;
      idx2_b += tile;
    }
    if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
    if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
    if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
    if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
  } else {
    y -= subm1;
    if (y >= x) y++;
    const int i_a = warp_a + y * tile;
    const int i_b = warp_b + y * tile;

    const int j_a = lane_a + x * tile;
    const int j_b = lane_b + x * tile;

    const int idx0_aa = i_a * upper + j_a;
    const int idx0_ab = i_a * upper + j_b;
    const int idx0_ba = i_b * upper + j_a;
    const int idx0_bb = i_b * upper + j_b;

    const int idx1_aa = lane_a * tile + warp_a;
    const int idx1_ab = lane_b * tile + warp_a;
    const int idx1_ba = lane_a * tile + warp_b;
    const int idx1_bb = lane_b * tile + warp_b;

    int idx2_a = (x * tile) * upper + j_a;
    int idx2_b = (x * tile) * upper + j_b;

    mtype ij_aa = AdjMat[idx0_aa];
    mtype ij_ab = AdjMat[idx0_ab];
    mtype ij_ba = AdjMat[idx0_ba];
    mtype ij_bb = AdjMat[idx0_bb];

    const mtype orig_aa = ij_aa;
    const mtype orig_ab = ij_ab;
    const mtype orig_ba = ij_ba;
    const mtype orig_bb = ij_bb;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      mtype ik_a, ik_b;
      if (k < ws) {
        ik_a = __shfl_sync(~0, ij_aa, k);
        ik_b = __shfl_sync(~0, ij_ba, k);
      }
      if (k >= ws) {
        ik_a = __shfl_sync(~0, ij_ab, k - ws);
        ik_b = __shfl_sync(~0, ij_bb, k - ws);
      }
      const mtype kr_a = krows[idx2_a];
      const mtype kr_b = krows[idx2_b];

      ij_aa = min(ij_aa, ik_a + kr_a);
      ij_ab = min(ij_ab, ik_a + kr_b);
      ij_ba = min(ij_ba, ik_b + kr_a);
      ij_bb = min(ij_bb, ik_b + kr_b);

      if (lane_a == k) temp[idx1_aa] = ij_aa;
      if (lane_a == k) temp[idx1_ba] = ij_ba;
      if (lane_b == k) temp[idx1_ab] = ij_ab;
      if (lane_b == k) temp[idx1_bb] = ij_bb;

      idx2_a += upper;
      idx2_b += upper;
    }
    __syncthreads();

    kcols[idx0_aa] = temp[warp_a * tile + lane_a];
    kcols[idx0_ab] = temp[warp_a * tile + lane_b];
    kcols[idx0_ba] = temp[warp_b * tile + lane_a];
    kcols[idx0_bb] = temp[warp_b * tile + lane_b];

    if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
    if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
    if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
    if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
  }
}

// calculate the remaining tiles
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void FWrem_64(mtype* const __restrict__ AdjMat, const int upper, mtype* const __restrict__ krows, mtype* const __restrict__ kcols, const int x, const int subm1)
{
  int y = blockIdx.x / subm1;
  int z = blockIdx.x % subm1;
  if (y >= x) y++;
  if (z >= x) z++;

  const int warp_a = threadIdx.x / ws;
  const int warp_b = warp_a + ws;

  const int lane_a = threadIdx.x % ws;
  const int lane_b = lane_a + ws;

  const int i_a = warp_a + y * tile;
  const int i_b = warp_b + y * tile;
  const int j_a = lane_a + z * tile;
  const int j_b = lane_b + z * tile;

  const int idx0_aa = i_a * upper + j_a; // upper left
  const int idx0_ab = i_a * upper + j_b; // upper right
  const int idx0_ba = i_b * upper + j_a; // lower left
  const int idx0_bb = i_b * upper + j_b; // lower right

  __shared__ mtype s_kj[tile * tile];
  __shared__ mtype s_ik[tile * tile];

  s_kj[warp_a * tile + lane_a] = krows[(x * tile + warp_a) * upper + j_a];
  s_kj[warp_a * tile + lane_b] = krows[(x * tile + warp_a) * upper + j_b];
  s_kj[warp_b * tile + lane_a] = krows[(x * tile + warp_b) * upper + j_a];
  s_kj[warp_b * tile + lane_b] = krows[(x * tile + warp_b) * upper + j_b];

  s_ik[warp_a * tile + lane_a] = kcols[i_a * upper + lane_a + x * tile];
  s_ik[warp_a * tile + lane_b] = kcols[i_a * upper + lane_b + x * tile];
  s_ik[warp_b * tile + lane_a] = kcols[i_b * upper + lane_a + x * tile];
  s_ik[warp_b * tile + lane_b] = kcols[i_b * upper + lane_b + x * tile];

  mtype ij_aa = AdjMat[idx0_aa];
  mtype ij_ab = AdjMat[idx0_ab];
  mtype ij_ba = AdjMat[idx0_ba];
  mtype ij_bb = AdjMat[idx0_bb];

  const mtype orig_aa = ij_aa;
  const mtype orig_ab = ij_ab;
  const mtype orig_ba = ij_ba;
  const mtype orig_bb = ij_bb;

  __syncthreads();
  int idx1_a = warp_a;
  int idx1_b = warp_b;

  int idx2_a = lane_a;
  int idx2_b = lane_b;

  #pragma unroll 64
  for (int k = 0; k < tile; k++) {
    const mtype sk_a = s_kj[idx2_a];
    const mtype sk_b = s_kj[idx2_b];

    ij_aa = min(ij_aa, s_ik[idx1_a] + sk_a);
    ij_ab = min(ij_ab, s_ik[idx1_a] + sk_b);
    ij_ba = min(ij_ba, s_ik[idx1_b] + sk_a);
    ij_bb = min(ij_bb, s_ik[idx1_b] + sk_b);

    idx1_a += tile;
    idx1_b += tile;

    idx2_a += tile;
    idx2_b += tile;
  }

  if ((y == z) && (y == x + 1) && (x != subm1)) { // the diagonal in next iteration
    const int idx1_aa = lane_a * tile + warp_a;
    const int idx1_ab = lane_b * tile + warp_a;
    const int idx1_ba = lane_a * tile + warp_b;
    const int idx1_bb = lane_b * tile + warp_b;

    int idx2_a = lane_a;
    int idx2_b = lane_b;

    #pragma unroll 64
    for (int k = 0; k < tile; k++) {
      if (warp_a == k) s_kj[idx2_a] = ij_aa;
      if (warp_a == k) s_kj[idx2_b] = ij_ab;
      if (warp_b == k) s_kj[idx2_a] = ij_ba;
      if (warp_b == k) s_kj[idx2_b] = ij_bb;
      __syncthreads();

      mtype ik_a, ik_b;
      if (k < ws) {
        ik_a = __shfl_sync(~0, ij_aa, k);
        ik_b = __shfl_sync(~0, ij_ba, k);
      }
      else {
        ik_a = __shfl_sync(~0, ij_ab, k - ws);
        ik_b = __shfl_sync(~0, ij_bb, k - ws);
      }

      const mtype sk_a = s_kj[idx2_a];
      const mtype sk_b = s_kj[idx2_b];

      ij_aa = min(ij_aa, ik_a + sk_a);
      ij_ab = min(ij_ab, ik_a + sk_b);
      ij_ba = min(ij_ba, ik_b + sk_a);
      ij_bb = min(ij_bb, ik_b + sk_b);

      if (warp_a == k) krows[idx0_aa] = ij_aa;
      if (warp_a == k) krows[idx0_ab] = ij_ab;
      if (warp_b == k) krows[idx0_ba] = ij_ba;
      if (warp_b == k) krows[idx0_bb] = ij_bb;

      if (lane_a == k) s_ik[idx1_aa] = ij_aa;
      if (lane_a == k) s_ik[idx1_ba] = ij_ba;
      if (lane_b == k) s_ik[idx1_ab] = ij_ab;
      if (lane_b == k) s_ik[idx1_bb] = ij_bb;
      idx2_a += tile;
      idx2_b += tile;
    }
    __syncthreads();

    kcols[idx0_aa] = s_ik[warp_a * tile + lane_a];
    kcols[idx0_ab] = s_ik[warp_a * tile + lane_b];
    kcols[idx0_ba] = s_ik[warp_b * tile + lane_a];
    kcols[idx0_bb] = s_ik[warp_b * tile + lane_b];
  }

  if (ij_aa != orig_aa) AdjMat[idx0_aa] = ij_aa;
  if (ij_ab != orig_ab) AdjMat[idx0_ab] = ij_ab;
  if (ij_ba != orig_ba) AdjMat[idx0_ba] = ij_ba;
  if (ij_bb != orig_bb) AdjMat[idx0_bb] = ij_bb;
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA ERROR %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

static void FW_gpu_64(const ECLgraph g, mtype* const AdjMat)
{
  // check device
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: there is no CUDA capable device\n\n"); exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  // printf("running on %s GPU with %d SMs (%.1f MHz core and %.1f MHz mem)\n", deviceProp.name, SMs, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  // copy graph to GPU
  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
  cudaMalloc((void **)&d_g.eweight, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.eweight, g.eweight, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  // allocate GPU memory
  const int sub = (g.nodes + tile - 1) / tile;

  const int upper = sub * tile; // upper bound of the GPU matrix
  mtype* d_AdjMat;
  if (cudaSuccess != cudaMalloc((void **)&d_AdjMat, sizeof(mtype) * upper * upper)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  mtype* d_krows;
  if (cudaSuccess != cudaMalloc((void **)&d_krows, sizeof(mtype) * upper * upper)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  mtype* d_kcols;
  if (cudaSuccess != cudaMalloc((void **)&d_kcols, sizeof(mtype) * upper * upper)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  // printf("GPU matrix size: %.1f MB\n", sizeof(mtype) * upper * upper / (1024.0 * 1024.0));

  timeval start, end;
  gettimeofday(&start, NULL);

  // run GPU init code
  init1<<<(upper * upper + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(g.nodes, d_AdjMat, upper);
  init2<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_AdjMat, upper);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  const double inittime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  // printf("FW_64 gpu init time: %10.6f s\n", inittime);
  CheckCuda();

  const int subm1 = sub - 1;
  //printf("FW_64 sub = %d (%d, %d)\n", sub, 2 * subm1, subm1 * subm1);
  gettimeofday(&start, NULL);

  // compute 64*64 tile
  FW0_64<<<1, ThreadsPerBlock>>>(d_AdjMat, upper, d_krows, d_kcols);
  if (sub > 1) {
    for (int x = 0; x < sub; x++) {
      FWrowcol_64<<<2 * subm1, ThreadsPerBlock>>>(d_AdjMat, upper, d_krows, d_kcols, x, subm1);
      FWrem_64<<<subm1 * subm1, ThreadsPerBlock>>>(d_AdjMat, upper, d_krows, d_kcols, x, subm1);
    }
  }

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  const double comptime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  // printf("FW_64 gpu comp time: %10.6f s\n", comptime);
  printf("%.5f\n", comptime*1000.0);
  CheckCuda();

  // copy result back to CPU
  if (cudaSuccess != cudaMemcpy(AdjMat, d_AdjMat, sizeof(mtype) * upper * upper, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // clean up
  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);
  cudaFree(d_g.eweight);
  cudaFree(d_AdjMat);
  cudaFree(d_krows);
  cudaFree(d_kcols);
}

static void FW_cpu(const ECLgraph g, mtype* const AdjMat)
{
  timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < g.nodes; i++) {
    for (int j = 0; j < g.nodes; j++) {
      AdjMat[i * g.nodes + j] = ((i == j) ? 0 : (INT_MAX / 2));
    }
  }

  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int nei = g.nlist[j];
      AdjMat[i * g.nodes + nei] = g.eweight[j];
    }
  }

  gettimeofday(&end, NULL);
  const double inittime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  // printf("CPU init time: %10.6f s\n", inittime);

  gettimeofday(&start, NULL);

  for (int k = 0; k < g.nodes; k++) {
    for (int i = 0; i < g.nodes; i++) {
      for (int j = 0; j < g.nodes; j++) {
        if (AdjMat[i * g.nodes + j] > AdjMat[i * g.nodes + k] + AdjMat[k * g.nodes + j]) {
          AdjMat[i * g.nodes + j] = AdjMat[i * g.nodes + k] + AdjMat[k * g.nodes + j];
        }
      }
    }
  }

  gettimeofday(&end, NULL);
  const double comptime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  // printf("CPU comp time: %10.6f s\n", comptime);
}

int main(int argc, char* argv[])
{
  // printf("ECL-APSP v1.0 (%s)\n", __FILE__);
  // printf("Copyright 2021 Texas State University\n");
  // if (argc != 2) {fprintf(stderr, "USAGE: %s input_graph_name\n\n", argv[0]);  exit(-1);}

  // read input
  // ECLgraph g = readECLgraph(argv[1]);
  ECLgraph g = readECLgraph_simd2();
  // printf("input: %s\n", argv[1]);
  // printf("nodes: %d\n", g.nodes);
  // printf("edges: %d\n", g.edges);
  // if (g.eweight == NULL) {fprintf(stderr, "ERROR: input graph has no edge weights\n\n");  exit(-1);}

  // make all weights positive to avoid negative cycles
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
    }
  }

  // run GPU_64 code
  const int upper_64 = ((g.nodes + tile - 1) / tile) * tile;  // round up
  mtype* const AdjMat1 = new mtype [upper_64 * upper_64];
  FW_gpu_64(g, AdjMat1);

  // // run CPU code
  // mtype* const AdjMat2 = new mtype [g.nodes * g.nodes];
  // FW_cpu(g, AdjMat2);

  // // compare results
  // int diffcount = 0;
  // const int gn = g.nodes;
  // for (int i = 0; i < gn; ++i) {
  //   for (int j = 0; j < gn; ++j) {
  //     if (AdjMat1[i * upper_64 + j] != AdjMat2[i * g.nodes + j]) {
  //       diffcount++;
  //       //printf("%d, %d\n", i, j);
  //     }
  //   }
  // }

  // if (diffcount > 0) {
  //   printf("ERROR: results differ in %d places!\n", diffcount);
  // } else {
  //   printf("results match\n");
  // }

  // clean up
  delete [] AdjMat1;
  // delete [] AdjMat2;
  freeECLgraph(g);
  return 0;
}
