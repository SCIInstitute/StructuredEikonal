//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <vector>

#include <iostream>
using namespace std;

// includes, project
//#include <cutil.h>

// common header
#include "common_def.h"

// includes, kernels
#include "cuda_fim_kernel.cu"

//#define DEBUG

#define TIMER

void CUT_SAFE_CALL(cudaError_t error) {
	if(error != cudaSuccess)
	{
		printf("CUDA error! %d \n",error);
		exit(EXIT_FAILURE);
	}
}

void CUDA_SAFE_CALL(cudaError_t error) {
	CUT_SAFE_CALL(error);
}

void runEikonalSolverSimple(CUDAMEMSTRUCT &cmem)
{
  int deviceID;
  cudaGetDevice(&deviceID);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceID);
  printf("Current device id : %d, name : %s\n", deviceID, deviceProp.name);

  int xdim, ydim, zdim;
  xdim = cmem.xdim;
  ydim = cmem.ydim;
  zdim = cmem.zdim;

#ifdef __DEVICE_EMULATION__
  assert(xdim%BLOCK_LENGTH == 0);
  assert(ydim%BLOCK_LENGTH == 0);
  assert(zdim%BLOCK_LENGTH == 0);
#endif

  // create volumes
  uint volSize = cmem.volsize;
  uint blockNum = cmem.blknum;

  //int nBlkZ = zdim/BLOCK_LENGTH;

  printf("# of total voxels : %d\n", volSize);
  printf("# of total blocks : %d\n", blockNum);

  // h_ : host memory, d_ : device memory

  // copy speed table to constant variable
  //cudaMemcpyToSymbol((const char*)_speed, cmem.h_speedtable, sizeof(float)*256);

  int nIter = cmem.nIter;
  uint nActiveBlock = cmem.nActiveBlock; // active list

  double *d_spd = cmem.d_spd;
  DOUBLE *d_sol = cmem.d_sol;
  DOUBLE *t_sol = cmem.t_sol;

  uint *d_list = cmem.d_list;
  bool *d_listVol = cmem.d_listVol;

  bool *d_con = cmem.d_con;
  bool *d_mask = cmem.d_mask;

  // copy so that original value should not be modified
  uint *h_list = (uint*) malloc(blockNum*sizeof(uint));
  bool *h_listed = (bool*) malloc(blockNum*sizeof(bool));
  bool *h_listVol = (bool*) malloc(blockNum*sizeof(bool));

  // initialization
  memcpy(h_list, cmem.h_list, blockNum*sizeof(uint));
  memcpy(h_listed, cmem.h_listed, blockNum*sizeof(bool));
  memcpy(h_listVol, cmem.h_listVol, blockNum*sizeof(bool));

  CUT_SAFE_CALL( cudaMemcpy(cmem.d_list, cmem.h_list, nActiveBlock*sizeof(uint), cudaMemcpyHostToDevice) );
  CUT_SAFE_CALL( cudaMemcpy(cmem.d_listVol, cmem.h_listVol, blockNum*sizeof(bool), cudaMemcpyHostToDevice) );
  CUT_SAFE_CALL( cudaMemcpy(cmem.d_sol, cmem.h_sol, volSize*sizeof(DOUBLE), cudaMemcpyHostToDevice) );
  CUT_SAFE_CALL( cudaMemcpy(cmem.t_sol, cmem.h_sol, volSize*sizeof(DOUBLE), cudaMemcpyHostToDevice) );
  CUT_SAFE_CALL( cudaMemset(cmem.d_con, 1, volSize*sizeof(bool)) );

  // set dimension of block and entire grid size
  dim3 dimBlock(BLOCK_LENGTH,BLOCK_LENGTH,BLOCK_LENGTH);
  dim3 dimEntireGrid(blockNum);
  dim3 dimGrid(nActiveBlock);

  int nTotalIter = 0;
  //uint sharedmemsize = sizeof(float)*BLOCK_LENGTH*BLOCK_LENGTH*(3*BLOCK_LENGTH + 2);

  vector<int> sourceList;
  sourceList.push_back((zdim/2)*ydim*xdim + (ydim/2)*xdim + (xdim/2));

  // initialize & start timer
  StopWatchInterface *timer_total, *timer_solver, *timer_reduction, *timer_list, *timer_list2, *timer_coarse;
  timer_total = timer_solver = timer_reduction = timer_list = timer_list2 = timer_coarse = NULL;

  sdkCreateTimer(&timer_total);
  sdkCreateTimer(&timer_solver);
  sdkCreateTimer(&timer_reduction);
  sdkCreateTimer(&timer_list);
  sdkCreateTimer(&timer_list2);
  sdkCreateTimer(&timer_coarse);

  sdkStartTimer(&timer_total);

  uint nTotalBlockProcessed = 0;

  // start solver
  while(nActiveBlock > 0)
  {
    //CUT_SAFE_CALL( cudaMemcpy(cmem.d_sol, cmem.h_sol, 64*nActiveBlock*sizeof(DOUBLE), cudaMemcpyHostToDevice) );
    assert(nActiveBlock < 4294967295);

    nTotalBlockProcessed += nActiveBlock;

    nTotalIter++;

    //
    // solve current blocks in the active lists
    //

    //      printf("# of active tiles : %u\n", nActiveBlock);
#ifdef DEBUG
    printf("# of active tiles : %u\n", nActiveBlock);
#endif
    //////////////////////////////////////////////////////////////////
    // 1. run solver on current active tiles

    sdkStartTimer(&timer_solver);

    dimGrid.y = (unsigned int)floor(((double)nActiveBlock-1)/65535)+1;
    dimGrid.x = (unsigned int)ceil ((double)nActiveBlock/(double)dimGrid.y);

#ifdef DEBUG
    printf("Grid size : %d x %d\n", dimGrid.x, dimGrid.y);
#endif

    CUT_SAFE_CALL( cudaMemcpy(d_list, h_list, nActiveBlock*sizeof(uint), cudaMemcpyHostToDevice) );

    run_solver<<< dimGrid, dimBlock >>>(d_spd, d_mask, d_sol, t_sol, d_con, d_list, xdim, ydim, zdim, nIter, nActiveBlock);

    CUT_SAFE_CALL(cudaGetLastError());

    cudaThreadSynchronize();
    sdkStopTimer(&timer_solver);


    //////////////////////////////////////////////////////////////////
    // 2. reduction (only active tiles)

    sdkStartTimer(&timer_reduction);

    run_reduction<<< dimGrid, dim3(BLOCK_LENGTH,BLOCK_LENGTH,BLOCK_LENGTH/2) >>>(d_con, d_listVol, d_list, nActiveBlock);

    CUT_SAFE_CALL(cudaGetLastError());
    //CUT_CHECK_ERROR("Kernel execution failed");
    cudaThreadSynchronize();

    sdkStopTimer(&timer_reduction);


    //////////////////////////////////////////////////////////////////
    // 3. check neighbor tiles of converged tile
    // Add any active block of neighbor of converged block is inserted
    // to the list

    // read-back active list volume
#ifdef TIMER
    sdkStartTimer(&timer_list);
#endif
    CUT_SAFE_CALL( cudaMemcpy(h_listVol, d_listVol, blockNum*sizeof(bool), cudaMemcpyDeviceToHost) );

    uint nOldActiveBlock = nActiveBlock;
    uint nBlkX = xdim/BLOCK_LENGTH;
    uint nBlkY = ydim/BLOCK_LENGTH;

    for(uint i=0; i<nOldActiveBlock; i++)
    {
      // check 6-neighbor of current active tile
      uint currBlkIdx = h_list[i];

      if(!h_listVol[currBlkIdx]) // not active : converged
      {
        uint nb[6];
        nb[0] = (currBlkIdx < nBlkX*nBlkY) ? currBlkIdx : (currBlkIdx - nBlkX*nBlkY);  //tp
        nb[1] = ((currBlkIdx + nBlkX*nBlkY) >= blockNum) ? currBlkIdx : (currBlkIdx + nBlkX*nBlkY); //bt
        nb[2] = (currBlkIdx < nBlkX) ? currBlkIdx : (currBlkIdx - nBlkX); //up
        nb[3] = ((currBlkIdx + nBlkX) >= blockNum) ? currBlkIdx : (currBlkIdx + nBlkX); //dn
        nb[4] = (currBlkIdx%nBlkX == 0) ? currBlkIdx : currBlkIdx-1; //lf
        nb[5] = ((currBlkIdx+1)%nBlkX == 0) ? currBlkIdx : currBlkIdx+1; //rt

        for(int nbIdx = 0; nbIdx < 6; nbIdx++)
        {
          uint currIdx = nb[nbIdx];

          //  assert(currIdx < volSize);

          if(!h_listed[currIdx])
          {
            h_listed[currIdx] = true;
            h_list[nActiveBlock++] = currIdx;
          }
        }
      }
    }
    cudaThreadSynchronize();
#ifdef TIMER
    sdkStopTimer(&timer_list);
#endif


    //////////////////////////////////////////////////////////////////
    // 4. run solver only once for neighbor blocks of converged block
    // current active list contains active blocks and neighbor blocks of
    // any converged blocks.
    //

#ifdef TIMER
    sdkStartTimer(&timer_solver);
#endif

    // update grid dimension because nActiveBlock is changed
    dimGrid.y = (unsigned int)floor(((double)nActiveBlock-1)/65535)+1;
    dimGrid.x = (unsigned int)ceil((double)nActiveBlock/(double)dimGrid.y);

#ifdef DEBUG
    printf("Grid size : %d x %d\n", dimGrid.x, dimGrid.y);
#endif

    CUT_SAFE_CALL(cudaMemcpy(d_list, h_list, nActiveBlock*sizeof(uint), cudaMemcpyHostToDevice) );
    run_check_neighbor<<< dimGrid, dimBlock >>>(d_spd, d_mask, t_sol, d_sol, d_con, d_list, xdim, ydim, zdim, nOldActiveBlock, nActiveBlock);
    CUT_SAFE_CALL(cudaGetLastError());
    cudaThreadSynchronize();

#ifdef TIMER
    sdkStopTimer(&timer_solver);
#endif



    //////////////////////////////////////////////////////////////////
    // 5. reduction

    sdkStartTimer(&timer_reduction);

    run_reduction<<< dimGrid, dim3(BLOCK_LENGTH,BLOCK_LENGTH,BLOCK_LENGTH/2) >>>(d_con, d_listVol, d_list, nActiveBlock);
    CUT_SAFE_CALL(cudaGetLastError());
    cudaThreadSynchronize();

    sdkStopTimer(&timer_reduction);


    //////////////////////////////////////////////////////////////////
    // 6. update active list
    // read back active volume from the device and add
    // active block to active list on the host memory


#ifdef TIMER
    sdkStartTimer(&timer_list2);
#endif
    nActiveBlock = 0;
    CUT_SAFE_CALL( cudaMemcpy(h_listVol, d_listVol, blockNum*sizeof(bool), cudaMemcpyDeviceToHost) );

    for(uint i=0; i<blockNum; i++)
    {
      if(h_listVol[i]) // true : active block (not converged)
      {
        h_listed[i] = true;
        h_list[nActiveBlock++] = i;
        //printf("Block %d added\n", i);
      }
      else h_listed[i] = false;
    }
    cudaThreadSynchronize();
#ifdef TIMER
    sdkStopTimer(&timer_list2);
#endif


#ifdef DEBUG
    printf("Iteration : %d\n", nTotalIter);
#endif
  }
  sdkStopTimer(&timer_total);

  printf("Eikonal solver converged after %d iterations\n", nTotalIter);
  printf("Total Running Time: %f (sec)\n", sdkGetTimerValue(&timer_total)/1000);
  printf("Time for solver : %f (sec)\n", sdkGetTimerValue(&timer_solver)/1000);
  printf("Time for reduction : %f (sec)\n", sdkGetTimerValue(&timer_reduction)/1000);
#ifdef TIMER
  printf("Time for list update-1 (CPU) : %f (sec)\n", sdkGetTimerValue(&timer_list)/1000);
  printf("Time for list update-2 (CPU) : %f (sec)\n", sdkGetTimerValue(&timer_list2)/1000);

#endif
  printf("Total # of blocks processed : %d\n", nTotalBlockProcessed);

  sdkDeleteTimer(&timer_total);
  sdkDeleteTimer(&timer_solver);
  sdkDeleteTimer(&timer_reduction);
  sdkDeleteTimer(&timer_list);

  // delete dynamically allocated host memory
  free(h_list);
  free(h_listed);
  free(h_listVol);
}
