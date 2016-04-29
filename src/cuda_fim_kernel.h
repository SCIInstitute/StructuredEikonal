//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

#ifndef _cuda_fim_KERNEL_H_
#define _cuda_fim_KERNEL_H_

#include <cstdio>
#include "common_def.h"

// check bank confilct only when device emulation mode
#ifdef __DEVICE_EMULATION__
#define CHECK_BANK_CONFLICTS
#endif

#ifdef CHECK_BANK_CONFLICTS
#define MEM(index) CUT_BANK_CHECKER(_mem, index)
#define SOL(i,j,k) CUT_BANK_CHECKER(((float*)&_sol[0][0][0]), (k*(BLOCK_LENGTH+2)*(BLOCK_LENGTH+2) + j*(BLOCK_LENGTH+2) + i))
#define SPD(i,j,k) CUT_BANK_CHECKER(((float*)&_spd[0][0][0]), (k*(BLOCK_LENGTH)*(BLOCK_LENGTH) + j*(BLOCK_LENGTH) + i))
#else
#define MEM(index) _mem[index]
#define SOL(i,j,k) _sol[i][j][k]
#define SPD(i,j,k) _spd[i][j][k]
#endif

__device__ DOUBLE get_time_eikonal(DOUBLE a, DOUBLE b, DOUBLE c, DOUBLE s);
//
// F : Input speed (positive)
// if F =< 0, skip that pixel (masking out)
//
__global__ void run_solver(double* spd, bool* mask, const DOUBLE *sol_in, 
  DOUBLE *sol_out, bool *con, uint* list, int xdim, int ydim, int zdim,
  int nIter, uint nActiveBlock);
//
// run_reduction
//
// con is pixelwise convergence. Do reduction on active tiles and write tile-wise
// convergence to listVol. The implementation assumes that the block size is 4x4x4.
//
__global__ void run_reduction(bool *con, bool *listVol, uint *list, uint nActiveBlock);
//
// if block is active block, copy values
// if block is neighbor, run solver once
//
__global__ void run_check_neighbor(double* spd, bool* mask, const DOUBLE *sol_in, DOUBLE *sol_out,
  bool *con, uint* list, int xdim, int ydim, int zdim,
  uint nActiveBlock, uint nTotalBlock);
#endif // #ifndef _cuda_fim_KERNEL_H_

