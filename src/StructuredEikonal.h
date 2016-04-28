#ifndef __STRUCTUREDEIKONAL_H__
#define __STRUCTUREDEIKONAL_H__

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include "common_def.h"
#include "cuda_fim.cu"

/** The class that represents all of the available options for StructuredEikonal */
class StructuredEikonal {
public:
  StructuredEikonal(bool verbose = false);
  virtual ~StructuredEikonal();
  void setDims(size_t w, size_t h, size_t d);
  void setMapType(size_t t);
  void setItersPerBlock(size_t t);
  void setSpeeds(std::vector<std::vector<std::vector<double> > > speed);
  void setSeeds(std::vector<std::array<size_t, 3> > seeds);
  void writeNRRD(std::string filename);
  std::vector< std::vector< std::vector<double> > > getFinalResult();
  /**
  * Runs the algorithm.
  *
  * @data The set of options for the Eikonal algorithm.
  *       The defaults are used if nothing is provided.
  */
  void solveEikonal();
  //public member for answer
  std::vector<std::vector<std::vector<double> > > answer_;
private:
  void error(char* msg);
  void CheckCUDAMemory();
  void init_cuda_mem();
  void set_attribute_mask();
  void initialization();
  void map_generator();
  void get_solution();
  void useSeeds();
  //data
  bool verbose_;
  bool isCudaMemCreated_;
  size_t width_, height_, depth_;
  size_t itersPerBlock_, solverType_;
  std::vector<std::vector<std::vector<double> > > speeds_;
  std::vector<std::array<size_t, 3> > seeds_;
  CUDAMEMSTRUCT memoryStruct_;
};

#endif
