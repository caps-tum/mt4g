# mt4g -- Memory Topology for GPUs

This project tries to capture the memory topology of Nvidia GPUs, specifically all GPUs since the Kepler microarchitecture.

We plan to add support for AMD GPUs as well as maintain the support for coming Nvidia microarchitectures.

## Pre-requisites & Compatibility

- cmake 3.22 or higher
- nvcc 11.0 or higher (tested on 11.4, 11.6 & 11.7)
- Tested on Nvidia Kepler, Maxwell, Pascal, Volta, Turing, Ampere GPUs.
- Primarilly developed for Linux, but should also work on Windows (however, not actively tested there).

## Installation

Cmake is the primary means for installing mt4g.

```bash
# pull latest version of cuda-samples git submodule
git submodule update --init

# build and install the benchmarking executable for a specific microarchitecture
mkdir build && cd build
cmake -DNVIDIA_UARCH="<version>" ..
# build options:
# -DNVIDIA_UARCH options are: KEPLER, MAXWELL, PASCAL, VOLTA, TURING, AMPERE
# -DIsDebug=1                           - turns on debug output
# -DCMAKE_INSTALL_PREFIX=../inst-dir    - to install locally into the git repo folder
make all install
```
## Running the Benchmarks

Run the benchmarsk with

```
../inst-dir/mt4g
# options:
#   -p:<path>:
#   	Overwrites the source of information for the number of Cuda Cores
# 	<path> specifies the path to the directory, that contains the 'deviceQuery' executable
#   -p: Overwrites the source of information for the number of Cuda Cores, uses 'nvidia-settings'
#   -d:<id> Sets the device, that will be benchmarked
#   -l1: Turns on benchmark for l1 data cache
#   -l2: Turns on benchmark for l2 data cache
#   -txt: Turns on benchmark for texture cache
#   -ro: Turns on benchmark for read-only cache
#   -c: Turns on benchmark for constant cache
```
If multiple GPUs are installed, you need to compile to the correct compute capability and then specify the device ID with the flag -d:
- `./mt4g -d:1` executes the tool using the GPU with deviceID 1
You can obtain the device ID again by calling `nvidia-smi` and checking the _GPU_ flag.

When the benchmarks are over (usually in 5-15 min), the final output is stored in "`GPU_Memory_Topology.csv`".

## Known issues/limitations

- For pascal microarchitecture, the process may evaluate L1 cache as not present (output in stdout "[L1_L2_DIFF.CUH]: Compare average values: L1 242.530000 <<>> L2 242.560000, compute absolute distance: 0.030000", where the L1 and L2 values are very similar) -- try building mt4g with a makefile instead (`make`).
- On Volta and Ampere, the L1/Texture/Readonly cache size is measured about 6–8 KiB less than the actual value (32KiB).

## About

mt4g has been initially created by Dominik Groessler (ge69qux@mytum.de) and the [CAPS TUM](https://www.ce.cit.tum.de/en/caps/homepage/), and is further maintained by Stepan Vanecek (stepan.vanecek@tum.de)  and the CAPS TUM. Please contact us in case of questions, bug reporting etc.

mt4g is available under the Apache-2.0 license. (see [License](https://github.com/caps-tum/mt4g/blob/master/LICENSE))
