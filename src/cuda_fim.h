//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//
#ifndef __CUDA_FIM_H__
#define __CUDA_FIM_H__

#include <cstdlib>
#include "common_def.h"

#define TIMER

void CUT_SAFE_CALL(cudaError_t error);
void CUDA_SAFE_CALL(cudaError_t error);
void runEikonalSolverSimple(CUDAMEMSTRUCT &cmem, bool verbose);

#endif