# GPUplayground

This project tries to capture the memory topology of Nvidia GPUs, specifically all GPUs since the Kepler microarchitecture.

## Compatibility:
- Tested on CUDA SDK 11.4, 11.6 & 11.7
- Tested on Nvidia Kepler, Maxwell, Pascal, Volta, Turing, Ampere GPUs

## Pre-requisites:
- Download and Install the CUDA SDK from here: https://developer.nvidia.com/cuda-downloads
- Clone the repository: `git clone`
- Fetch the submodule: `git submodule --update --init`

## Build:
- `make` for building without debug output
- `make debug` for additional debug output (files)
- The repository has a makefile and a cmakefile, which can be used after the CUDA installation to build the project.
- It may be necessary to adjust the makefile or CMakeLists.txt according to your GPU compute capability
  * In the makefile this can be done in the CXXARGS by specifying `-gencode arch=compute_xx,code=sm_xx`
  * In the CMakeLists.txt this can be done via setting _COMPUTE_CAP_COMPILE_OPTIONS_, examples for that are commented in the CMakeLists.txt
- You can check the compute capability of your GPU by calling `nvidia-smi`

## Execution:
- Linux: `./MemTop`
- Windows: `MemTop.exe`

If multiple GPUs are installed, you need to compile to the correct compute capability and then specify the device ID with the flag -d:
- `MemTop -d:1` executes the tool using the GPU with deviceID 1

You can obtain the device ID again by calling `nvidia-smi` and checking the _GPU_ flag
