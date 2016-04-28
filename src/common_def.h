//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

//
// Common to entire project
//

#ifndef __COMMON_DEF_H__
#define __COMMON_DEF_H__

#include "float.h"
#include "math.h"
#include "helper_timer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
//
// common definition for Eikonal solvers
//
#ifndef INF
#define INF 1e20//FLT_MAX //
#endif

#define BLOCK_LENGTH 4

#ifndef FLOAT

	#define DOUBLE double
	#define EPS (DOUBLE)1e-16

#else

	#define DOUBLE float
	#define EPS (DOUBLE)1e-6

#endif

//
// itk image volume definition for 3D anisotropic eikonal solvers
//
typedef unsigned int uint;
typedef unsigned char uchar;

struct CUDA_MEM_STRUCTURE {
  // volsize/blksize : # of pixel in volume/block
  // blknum : # of block
  // blklength : # of pixel in one dimemsion of block
  uint nActiveBlock, blknum, volsize, blksize;
  int xdim, ydim, zdim, nIter, blklength; // new new x,y,z dim to aligh power of 4

  // host memory
  uint *h_list;
  bool *h_listVol, *h_listed;

  // device memory
  uint *d_list;
  double *d_spd;
  bool *d_mask, *d_listVol, *d_con;	
  
  DOUBLE *h_sol;//h_speedtable[256];
  DOUBLE *d_sol, *t_sol; 

  // GroupOrder
  int* blockOrder;
  int K;
};

typedef struct CUDA_MEM_STRUCTURE CUDAMEMSTRUCT;

void CUT_SAFE_CALL(cudaError_t error);
void CUDA_SAFE_CALL(cudaError_t error);

#endif
