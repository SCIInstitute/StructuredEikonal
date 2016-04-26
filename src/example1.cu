//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// Modified by Sumin Hong (sumin246@unist.ac.kr)
//
// 2016. 2. 4
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include<vector>
#include "common_def.h"
#include "cuda_fim.cu"
#include <fstream>
#include <vector>

int iterperblock, runmode,  solvertype;
bool isCudaMemCreated;
double* speedF;

void writeNRRD(CUDAMEMSTRUCT & data, size_t xdim, size_t ydim, size_t zdim) {
  std::fstream out("test.nrrd", std::ios::out | std::ios::binary);
  out << "NRRD0001\n";
  out << "# Complete NRRD file format specification at:\n";
  out << "# http://teem.sourceforge.net/nrrd/format.html\n";
  out << "type: double\n";
  out << "dimension: 3\n";
  out << "sizes: " << xdim << " " << ydim << " " << zdim << "\n";
  out << "endian: little\n";
  out << "encoding: raw\n\n";
  for(int idx = 0; idx < xdim*ydim*zdim; idx++) {
    double d = data.h_sol[idx];
    out.write(reinterpret_cast<const char*>(&d),sizeof(double));
  }
  out.close();
}

void error(char* msg)
{
  printf("%s\n",msg);
  assert(false);
  exit(0);
}

void CheckCUDAMemory()
{
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << "Total Memory : " << totalMem/(1024*1024) << "MB" << std::endl;
  std::cout << "Free Memory  : " << freeMem/(1024*1024)  << "MB" << std::endl;
  std::cout << "--" << std::endl;
}

void init_cuda_mem(CUDAMEMSTRUCT &_mem, int xdim, int ydim, int zdim)
{
  assert(xdim > 0 && ydim > 0 && zdim > 0);
  if(xdim <= 0 || ydim <= 0 || zdim <= 0) printf("Volume dimension cannot be zero");

  CheckCUDAMemory();

  // 1. Create /initialize GPU memory
  int nx, ny, nz;

  nx = xdim + (BLOCK_LENGTH-xdim%BLOCK_LENGTH)%BLOCK_LENGTH;
  ny = ydim + (BLOCK_LENGTH-ydim%BLOCK_LENGTH)%BLOCK_LENGTH;
  nz = zdim + (BLOCK_LENGTH-zdim%BLOCK_LENGTH)%BLOCK_LENGTH;

  printf("%d %d %d \n",nx,ny,nz);

  uint volSize = nx*ny*nz;
  uint blkSize = BLOCK_LENGTH*BLOCK_LENGTH*BLOCK_LENGTH;

  int nBlkX = nx/BLOCK_LENGTH;
  int nBlkY = ny/BLOCK_LENGTH;
  int nBlkZ = nz/BLOCK_LENGTH;
  uint blockNum = nBlkX*nBlkY*nBlkZ;

  _mem.xdim = nx;
  _mem.ydim = ny;
  _mem.zdim = nz;
  _mem.volsize = volSize;
  _mem.blksize = blkSize;
  _mem.blklength = BLOCK_LENGTH;
  _mem.blknum = blockNum;
  _mem.nIter = iterperblock; // iter per block

  if(isCudaMemCreated) // delete previous memory
  {
    free((DOUBLE*)_mem.h_sol);
    free((uint*)_mem.h_list);
    free((bool*)_mem.h_listed);
    free((bool*)_mem.h_listVol);
    free((int*)_mem.blockOrder);
    CUDA_SAFE_CALL( cudaFree(_mem.d_spd) );
    CUDA_SAFE_CALL( cudaFree(_mem.d_sol) );
    CUDA_SAFE_CALL( cudaFree(_mem.t_sol) );  // temp solution for ping-pong
    CUDA_SAFE_CALL( cudaFree(_mem.d_con) );  // convergence volume
    CUDA_SAFE_CALL( cudaFree(_mem.d_list) );
    CUDA_SAFE_CALL( cudaFree(_mem.d_listVol) );
    CUDA_SAFE_CALL( cudaFree(_mem.d_mask) );
  }
  isCudaMemCreated = true;

  _mem.h_sol = (DOUBLE*) malloc(volSize*sizeof(DOUBLE)); // initial solution
  _mem.h_list = (uint*) malloc(blockNum*sizeof(uint)); // linear list contains active block indices
  _mem.h_listed = (bool*) malloc(blockNum*sizeof(bool));  // whether block is added to the list
  _mem.h_listVol = (bool*) malloc(blockNum*sizeof(bool)); // volume list shows active/nonactive of corresponding block
  _mem.blockOrder = (int*) malloc(blockNum*sizeof(int));

  CheckCUDAMemory();

  //
  // create host/device memory using CUDA mem functions
  //
  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.d_spd), volSize*sizeof(double)) );
  CheckCUDAMemory();

  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.d_sol), volSize*sizeof(DOUBLE)) );
  CheckCUDAMemory();

  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.t_sol), volSize*sizeof(DOUBLE)) );  // temp solution for ping-pong
  CheckCUDAMemory();

  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.d_con), volSize*sizeof(bool))  );  // convergence volume
  CheckCUDAMemory();

  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.d_list), blockNum*sizeof(uint)) );
  CheckCUDAMemory();

  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.d_listVol), blockNum*sizeof(bool)) );
  CheckCUDAMemory();

  CUDA_SAFE_CALL( cudaMalloc((void**)&(_mem.d_mask), volSize*sizeof(bool)) );
  CheckCUDAMemory();
}

void set_attribute_mask(CUDAMEMSTRUCT &_mem)
{
  uint volSize = _mem.volsize;

  int nx, ny, nz, blklength;

  nx = _mem.xdim;
  ny = _mem.ydim;
  nz = _mem.zdim;
  blklength = _mem.blklength;

  // create host memory
  double *h_spd  = new double[volSize]; // byte speed, host
  bool  *h_mask = new bool[volSize];

  // copy input volume to host memory
  // make each block to be stored contiguously in 1D memory space
  uint idx = 0;
  for(int zStr = 0; zStr < nz; zStr += blklength)
  {
    for(int yStr = 0; yStr < ny; yStr += blklength)
    {
      for(int xStr = 0; xStr < nx; xStr += blklength)
      {
        // for each block
        for(int z=zStr; z<zStr+blklength; z++)
        {
          for(int y=yStr; y<yStr+blklength; y++)
          {
            for(int x=xStr; x<xStr+blklength; x++)
            {
              uint midx = z*nx*ny + y*nx + x;
              h_spd[idx] = speedF[midx];
              //h_mask[idx] = speedF[midx] > 0 ? true : false;//false; // mask out
              h_mask[idx] = true;// : false;//false; // mask out
              idx++;
            }
          }
        }
      }
    }
  }

  // initialize GPU memory with host memory
  CUDA_SAFE_CALL( cudaMemcpy(_mem.d_spd, h_spd, volSize*sizeof(double), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(_mem.d_mask, h_mask, volSize*sizeof(bool), cudaMemcpyHostToDevice) );

  delete[] h_spd;
  delete[] h_mask;
}

void initialization(CUDAMEMSTRUCT &mem,int xdim, int ydim, int zdim)
{
  // get / set CUDA device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  int device;
  for(device = 0; device < deviceCount; ++device)
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
  }

  CheckCUDAMemory();

  init_cuda_mem(mem,xdim,ydim,zdim);

  set_attribute_mask(mem);

  CheckCUDAMemory();

}

void set_seed(CUDAMEMSTRUCT &_mem)
{
  //cout << "Loading seed volume..." << endl;
  uint volSize, blockNum;
  int nx, ny, nz, blklength;

  nx = _mem.xdim;
  ny = _mem.ydim;
  nz = _mem.zdim;
  volSize = _mem.volsize;
  blklength = _mem.blklength;

  blockNum = _mem.blknum;

  // copy input volume to host memory
  // make each block to be stored contiguously in 1D memory space
  uint idx = 0;
  uint blk_idx = 0;
  uint list_idx = 0;
  uint nActiveBlock = 0;

  for(int zStr = 0; zStr < nz; zStr += blklength)
  {
    for(int yStr = 0; yStr < ny; yStr += blklength)
    {
      for(int xStr = 0; xStr < nx; xStr += blklength)
      {
        // for each block
        bool isSeedBlock = false;

        for(int z=zStr; z<zStr+blklength; z++)
        {
          for(int y=yStr; y<yStr+blklength; y++)
          {
            for(int x=xStr; x<xStr+blklength; x++)
            {
              int seedVal;

              if (x == nx/2 && y == ny/2 && z == nz/2)
              {
                printf("%d,%d,%d is selected by source \n",x,y,z);
                seedVal = 1;
              }
              else
              {
                seedVal = 2;
              }

              _mem.h_sol[idx] = INF;

              if(seedVal == 1) // seed
              {
                _mem.h_sol[idx] = 0;
                isSeedBlock = true;
                //  printf("%d is Selected bt source \n",idx);
              }

              idx++;
            }
          }
        }

        ///////////////////////////////////////////////

        if(isSeedBlock)
        {
          //    printf("%d,%d,%d is Seed Block \n",zStr,yStr,xStr);

          _mem.h_listVol[blk_idx] = true;
          _mem.h_listed[blk_idx] = true;
          _mem.h_list[list_idx] = blk_idx;
          list_idx++;
          nActiveBlock++;
        }
        else
        {
          _mem.h_listVol[blk_idx] = false;
          _mem.h_listed[blk_idx] = false;
        }

        blk_idx++;
      }
    }
  }
  //  nActiveBlock = 1;

  _mem.nActiveBlock = nActiveBlock;

  // initialize GPU memory with host memory
  CUDA_SAFE_CALL( cudaMemcpy(_mem.d_sol, _mem.h_sol, volSize*sizeof(DOUBLE), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(_mem.t_sol, _mem.h_sol, volSize*sizeof(DOUBLE), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(_mem.d_list, _mem.h_list, nActiveBlock*sizeof(uint), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(_mem.d_listVol, _mem.h_listVol, blockNum*sizeof(bool), cudaMemcpyHostToDevice) );

  // initialize GPU memory with constant value
  CUDA_SAFE_CALL( cudaMemset(_mem.d_con, 1, volSize*sizeof(bool)) );
}

void get_solution(CUDAMEMSTRUCT &_mem)
{
  // copy solution from GPU
  CUDA_SAFE_CALL( cudaMemcpy(_mem.h_sol, _mem.d_sol, _mem.volsize*sizeof(DOUBLE), cudaMemcpyDeviceToHost) );
  //put the data where it belongs in the grand scheme of data!
  std::vector<double> real_data(_mem.volsize, 0);
  for(size_t blockID = 0; blockID < _mem.blknum; blockID++) {
    size_t baseAddr = blockID * _mem.blksize;
		size_t xgridlength = _mem.xdim/BLOCK_LENGTH;
		size_t ygridlength = _mem.ydim/BLOCK_LENGTH;
		// compute block index
		size_t bx = blockID%xgridlength;
		size_t tmpIdx = (blockID - bx)/xgridlength;
		size_t by = tmpIdx%ygridlength;
		size_t bz = (tmpIdx-by)/ygridlength;
    //translate back to real space
    for(int k = 0; k < BLOCK_LENGTH; k++) {
      for(int j = 0; j < BLOCK_LENGTH; j++) {
        for(int i = 0; i < BLOCK_LENGTH; i++) {
          double d = _mem.h_sol[baseAddr + 
            k * BLOCK_LENGTH * BLOCK_LENGTH + 
            j * BLOCK_LENGTH + i];
          size_t newIdx = (i + bx * BLOCK_LENGTH) * _mem.xdim * _mem.ydim + 
            (j + by * BLOCK_LENGTH)  * _mem.xdim + k + bz * BLOCK_LENGTH;
          if (newIdx < _mem.volsize) {
            real_data[newIdx] = d;
          }
        }
      }
    }
  }
  for(size_t i = 0; i < _mem.volsize; i++) {
    _mem.h_sol[i] = real_data[i];
  }
}

void map_generator(int type,int xdim,int ydim, int zdim)
{
  double pi = 3.141592653589793238462643383;

  speedF = new double[zdim * ydim * xdim];

  double* grid = speedF;

  for ( int i = 0 ; i < zdim * ydim * xdim ; i++)
    grid[i] = 1.0;

  switch(type){
  case 0 :
    //Constant Speed Map
    break;
  case 1 :
    //Sinusoid Speed Map
    for (int k = 0 ; k < zdim ; ++k)
      for (int j = 0 ; j < ydim ; ++j)
        for ( int i = 0 ; i < xdim ; ++i)
        {
          int idx = k*ydim*xdim  + j*xdim +i;
          grid[idx] = (6 + 5*(sin((i*pi)/xdim*2))*sin((j*pi)/ydim*2)*sin((k*pi)/zdim*2));
        }
    break;
  }
}

int main(int argc, char** argv) {
  int map_type = 0;
  int xdim, ydim, zdim;
  xdim = ydim = zdim = 256;
  iterperblock = 10;

  int i = 1;
  while (i < argc){
    if (strcmp(argv[i],"--help") == 0)
    {
      printf("Usage : %s -s [SIZE] -m [MapOption] \n",argv[0]);
      printf("MapType : 0 - Constant, 1 - sinusoid\n");
      return 0;
    }
    if (strcmp(argv[i],"-m") == 0)
      map_type = atoi(argv[i+1]);
    if (strcmp(argv[i],"-s") == 0)
      xdim = ydim = zdim = atoi(argv[i+1]);
    i+=2;
  }

  map_generator(map_type,xdim,ydim,zdim);

  CUDAMEMSTRUCT cmem;
  isCudaMemCreated = false;

  initialization(cmem,xdim,ydim,zdim);
  set_seed(cmem);

  runEikonalSolverSimple(cmem);

  get_solution(cmem);

  writeNRRD(cmem,xdim,ydim,zdim);

  return 0;
}