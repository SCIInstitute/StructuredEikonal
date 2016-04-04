GPUTUM: StructuredEikonal
=====================

GPUTUM: StructuredEikonal is a C++/CUDA library written to solve the Eikonal equation on structured meshes. It uses the fast iterative method (FIM) to solve efficiently, and uses GPU hardware.

The code was written by Won-Ki Jeong at the Scientific Computing and Imaging Institute, 
University of Utah, Salt Lake City, USA. The theory behind this code is published in the papers linked below. 
Table of Contents
========
- [Aknowledgements](#eikonal-3d-aknowledgements)
- [Requirements](#requirements)
- [Building](#building)<br/>
		- [Linux / OSX](#linux-and-osx)<br/>
		- [Windows](#windows)<br/>
- [Running Examples](#running-examples)
- [Using the Library](#using-the-library)
- [Testing](#testing)<br/>

<h4>Aknowledgements</h4>
**<a href ="http://people.seas.harvard.edu/~wkjeong/publication/wkjeong-sisc-fim.pdf">A Fast Iterative Method for Eikonal Equations</a>**<br/>
<img src=""  align="right" hspace="20" width=450>
*NOTE*<br/>

**AUTHORS:**
Won-Ki Jeong(*b*) <br/>
Ross T. Whitaker(*a*) <br/>

This library solves for the Eikional values on vertices located on a structured mesh.

<br/><br/>
Requirements
==============

 * Git, CMake (2.8+ recommended), and the standard system build environment tools.
 * You will need a CUDA Compatible Graphics card. See <a href="https://developer.nvidia.com/cuda-gpus">here</a> You will also need to be sure your card has CUDA compute capability of at least 2.0.
 * SCI-Solver_Eikonal is compatible with the latest CUDA toolkit (7.5). Download <a href="https://developer.nvidia.com/cuda-downloads">here</a>.
 * This project has been tested on OpenSuse 13.1 (Bottle) on NVidia GeForce GTX 680 HD, Windows 7 on NVidia GeForce GTX 775M, and OSX 10.10 on NVidia GeForce GTX 775M. 
 * If you have a CUDA graphics card equal to or greater than our test machines and are experiencing issues, please contact the repository owners.
 * Windows: You will need Microsoft Visual Studio 2010+ build tools. This document describes the "NMake" process.
 * OSX: Please be sure to follow setup for CUDA <a href="http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/#axzz3W4nXNNin">here</a>. There are several compatability requirements for different MAC machines, including using a different version of CUDA (ie. 5.5).

Building
==============

<h3>Linux and OSX</h3>
In a terminal:
```c++
mkdir StructuredEikonal/build
cd StructuredEikonal/build
cmake ../src
make
```

<h3>Windows</h3>
Open a Visual Studio (32 or 64 bit) Native Tools Command Prompt. 
Follow these commands:
```c++
mkdir C:\Path\To\StructuredEikonal\build
cd C:\Path\To\StructuredEikonal\build
cmake -G "NMake Makefiles" ..\src
nmake
```

**Note:** For all platforms, you may need to specify your CUDA toolkit location (especially if you have multiple CUDA versions installed):
```c++
cmake -DCUDA_TOOLKIT_ROOT_DIR="~/NVIDIA/CUDA-7.5" ../src
```
(Assuming this is the location).

**Note:** If you have compile errors such as <code>undefined reference: atomicAdd</code>, it is likely you need to set your compute capability manually. CMake outputs whether compute capability was determined automatically, or if you need to set it manually. The default minimum compute capability is 2.0.

```c++
cmake -DCUDA_COMPUTE_CAPABILITY=20 ../src
make
```

Running Examples
==============

Examples have not yet been provided.

Using the Library
==============

The library has not yet been implemented for generic usage.

Testing
==============

Testing has not yet been implemented.
