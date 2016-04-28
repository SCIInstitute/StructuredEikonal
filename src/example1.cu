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
  StructuredEikonal data;
  size_t itersPerBlock = 10, size = 256, type = 0;
  int i = 1;
  while (i < argc){
    if (strcmp(argv[i],"--help") == 0)
    {
      printf("Usage : %s -s [SIZE] -m [MapOption] -i [Iters per Block]\n",argv[0]);
      printf("MapType : 0 - Constant, 1 - sinusoid\n");
      return 0;
    }
    if (strcmp(argv[i],"-m") == 0)
      type = atoi(argv[i+1]);
    if (strcmp(argv[i],"-s") == 0)
      size = atoi(argv[i+1]);
    if (strcmp(argv[i],"-i") == 0)
      itersPerBlock = atoi(argv[i+1]);
    i+=2;
  }
  data.setDims(size,size,size);
  data.setMapType(type);
  data.setItersPerBlock(itersPerBlock);
  data.solveEikonal();
  data.writeNRRD("test.nrrd");
  return 0;
}