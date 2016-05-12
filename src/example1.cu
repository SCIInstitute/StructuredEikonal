//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// Modified by Sumin Hong (sumin246@unist.ac.kr)
//
// 2016. 2. 4
//

#include <StructuredEikonal.h>

int main(int argc, char** argv) {
  size_t itersPerBlock = 10, size = 256, type = 0;
  int i = 1;
  std::string name = "test.nrrd";
  bool verbose = false;
  while (i < argc) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      std::cout << "Usage : " << argv[0] << " [Options]" << std::endl;
      std::cout << "     -s SIZE              Volume size (cubed). [256]" << std::endl;
      std::cout << "     -m TYPE              Initialize speeds (constant [0], egg carton [1])." << std::endl;
      std::cout << "     -i ITER_PER_BLOCK    Number of iterations per block. [10]" << std::endl;
      std::cout << "     -o OUTPUT_NAME       Name of the output file. [test.nrrd]" << std::endl;
      std::cout << "     -v                   Verbose output." << std::endl;
      return 0;
    }
    if (strcmp(argv[i],"-m") == 0)
      type = atoi(argv[++i]);
    if (strcmp(argv[i],"-s") == 0)
      size = atoi(argv[++i]);
    if (strcmp(argv[i], "-o") == 0)
      name = argv[++i];
    if (strcmp(argv[i], "-v") == 0)
      verbose = true;
    if (strcmp(argv[i],"-i") == 0)
      itersPerBlock = atoi(argv[++i]);
    i++;
  }
  StructuredEikonal data(verbose);
  data.setDims(size,size,size);
  data.setMapType(type);
  data.setItersPerBlock(itersPerBlock);
  data.setSeeds({ { { { 0, 0, 0 } } } }); // set 0 0 0 voxel to zero
  data.solveEikonal();
  data.writeNRRD(name);
  return 0;
}