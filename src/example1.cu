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
    if (strcmp(argv[i],"--help") == 0) {
      std::cout << "Usage : " << argv[0] << "-s [SIZ(256)E] -m [MAP OPTION (0,1)]"
        << "-i [ITER_PER_BLOCK(10)] -o [OUTPUT_NAME(test.nrrd)] -v [verbose?(false)]\n";
      printf("MapType : 0 - Constant, 1 - sinusoid\n");
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