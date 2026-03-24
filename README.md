# mt4g - Memory Topology 4 GPUs

**mt4g** is a vendor-agnostic collection of microbenchmarks and APIs that
explores the compute and memory topologies of both AMD and NVIDIA GPUs based on
the HIP toolchain. By capturing system properties such as the number of SMs/CUs,
warp size, memory and cache sizes, cache line sizes and load latencies as well
as exposing deep cache subsystems and their physical layouts, it provides
critical support for GPU performance modeling and analysis within one unified
interface.

## Overview

The **mt4g** CLI tool enables a unified and cross-platform introspection of the
hardware topology of GPUs and thus provides crucial information that is either
scattered throughout vendor-specific APIs and data sheets or otherwise
programmatically unavailable. Key features include:

- Compilation of existing APIs and over 50 microbenchmarks for statistical
  topology attribute measurement
- Unified build system for AMD and NVIDIA targets
- Comprehensive report of collected benchmark results as structured JSON with
  optional plot generation for visualization

## Installation

### Dependencies

- ROCm or CUDA backend including drivers, compilers and libraries for AMD or
  NVIDIA targets respectively
- HIP SDK with the `hipcc` compiler
- `nlohmann-json` for JSON output
- `cxxopts` for CLI parsing
- Python 3 including the `matplotlib`, `pandas` and `numpy` packages for
  graphical plots

A suitable HIP environment can for instance be obtained via
[Spack](https://spack.readthedocs.io/en/latest/getting_started.html):

```bash
spack install hip           # includes ROCm backend for AMD targets
spack install hip+cuda      # includes CUDA backend for NVIDIA targets

spack load hip              # exports binaries and libraries
```

The `HIP_PATH` environment variable should be set to the HIP installation
directory. Please export manually if not automatically set by `spack`, e.g.

```bash
export HIP_PATH=<path_to_spack>/opt/spack/<system_architecture>/hip-<version>-<hash>
```

Additionally for NVIDIA targets, the `CUDA_PATH` environment variable needs to
be set to the CUDA installation directory.

**mt4g** has been tested successfully with `hip@6.3.3` and `cuda@12.8`.

### Build

Use the `GPU_TARGET_ARCH` build flag to select the target GPU architecture for
AMD (e.g. `gfx90a`) and NVIDIA (e.g. `sm_90`) respectively. Some of the
identifiers of the LLVM targets for AMD can be found
[here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus),
while the compute capabilites for NVIDIA can be found [here](https://developer.nvidia.com/cuda/gpus).
To build and install **mt4g**, run

```bash
git clone https://github.com/caps-tum/mt4g.git
cd mt4g
mkdir build && cd build
cmake .. -DGPU_TARGET_ARCH=<gfxXXX|sm_XX>
# optional build flags:
# -DCMAKE_BUILD_TYPE=<Release|Debug>             -- to choose between release and debug builds
# -DCMAKE_INSTALL_PREFIX=<install_prefix>        -- to set the install destination (default on UNIX platforms: /usr/local)
make all install -j $(nproc)
```

## Usage

```bash
<install_prefix>/bin/mt4g [options]
```

### Options

| Option | Description |
| ------ | ----------- |
| `-d, --device-id <id>` | GPU device to use (default `0`) |
| `-f, --file <name>` | Specify name of output files (default `<GPU_NAME>`) |
| `-g, --graphs` | Generate graphical plots for each benchmark |
| `-l, --location <path>` | Specify location of output files (default `.`) |
| `-o, --raw` | Write raw timing data |
| `-p, --report` | Create Markdown report in output directory |
| `-r, --random` | Randomize P-Chase arrays |
| `-s, --stdout` | Dump final JSON result into stdout |
| `-q, --quiet` | Only write the final JSON to stdout |
| `--l1` | Run L1 cache benchmarks |
| `--l2` | Run L2 cache benchmarks |
| `--l3` | Run L3 cache benchmarks (AMD only) |
| `--scalar` | Run AMD scalar cache benchmarks |
| `--constant` | Run NVIDIA constant cache benchmarks |
| `--readonly` | Run NVIDIA read-only cache benchmarks |
| `--texture` | Run NVIDIA texture cache benchmarks |
| `--shared` | Run shared memory benchmarks |
| `--memory` | Run main memory benchmarks |
| `--departuredelay` | Run departure delay benchmarks |
| `--resourceshare` | Run resource sharing benchmarks |
| `-v, --version` | Display the version of mt4g and exit |
| `-h, --help` | Display a detailed help message and exit |

If no benchmark group is chosen, all available groups are executed. Unsupported
groups are disabled automatically depending on the platform. Exclusive GPU
access is recommended for more reliable measurement results.

### Output

Usually, benchmark results are written as structured JSON into the file
`<GPU_NAME>.json` of the current working directory. However, the name and path
of the output file and directory may be changed through the flags `-f`/`--file`
and `-l`/`--location` respectively. With `-s`/`--stdout`, the final JSON output
file may be dumped into `stdout` instead. When `--graphs`, `--raw` or `--report`
is enabled, additional files are written to `results/<GPU_NAME>`. The `--report`
flag generates a `README.md` that embeds all graphs and links to the raw data.

If you would like to contribute results for hardware not yet covered, please
run the tool with `--raw --graphs --report` and send us the generated directory
via pull request.

## Supported Architectures

**mt4g** works reliably on all AMD CDNA GPUs and all recent NVIDIA
microarchitectures from Pascal onwards. However, the Ada Lovelace and Blackwell
architectures have not yet been tested due to missing access. Furthermore, we
do not consider AMD's RDNA GPUs since we focus on HPC/AI workloads. Support for
UDNA GPUs is planned in the future.

## Topological metrics

The information on the attributes provided by **mt4g** is depicted below

### AMD

| _Memory Element_ | Size | Load Latency | Read & Write Bandwidth | Cache Line Size | Fetch Granularity | Amount per SM/CU or GPU | Physically Shared With |
| ---------------- | ---- | ------------ | ---------------------- | --------------- | ----------------- | ----------------------- | ---------------------- |
| **vL1 cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **sL1d cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **L2 cache** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **L3 cache** | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **LDS** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Device Memory** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

### NVIDIA

| _Memory Element_ | Size | Load Latency | Read & Write Bandwidth | Cache Line Size | Fetch Granularity | Amount per SM/CU or GPU | Physically Shared With |
| ---------------- | ---- | ------------ | ---------------------- | --------------- | ----------------- | ----------------------- | ---------------------- |
| **L1 cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **L2 cache** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Texture cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Readonly cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Constant L1 cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Constant L1.5 cache** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Shared Memory** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Device Memory** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

## Known Issues and Limitations

- L2 segment size measurements on AMD GPUs are currently unreliable due to the platform's complex cache behaviour.
- Constant L1.5 Cache Size detection is capped at 64 KiB. Denoted by 64 KiB + 1 and confidence = 0. (> 64 KiB)
- Bandwidths are not optimal because we currently do not use a (dynamically found) optimal number of blocks.
- Cache Line Size detection uses a heuristical approach and is therefore not guaranteed to be correct.
- Constant L1 shared with L1 is not too reliable. Hence, as a hotfix we repeat the measurements 10 times and on one unsuccessful run return not shared. We are working on a cleaner solution.
- Runs only on Linux.

## Repository Layout & Contribution Guidelines

```
mt4g
├── CMakeLists.txt        -- Build configuration
├── include               -- Header files
├── LICENSE               -- Project license
├── README.md             -- Project description
├── sample_results        -- Exemplary output files from selected hardware
└── src                   -- Benchmark implementation and CLI helpers
```

### Adding new Measurements

Pre-measured results for selected GPUs live in the
[sample_results](sample_results/) directory. If your hardware is not yet listed,
we would greatly appreciate additional reports: Run the tool with
`--raw --graphs --report` and open a pull request to share your measurements.

### Adding a new Benchmark

To add a new benchmark to the **mt4g**, follow the subsequent instructions:

1. Implement the benchmark in `src/benchmarks/` and expose a suitable interface
   in `include/`.
2. Try to follow the pattern of `measureXXX()`, `XXXLauncher()` and `XXXKernel()` to keep the structure modular and readable.
   Every benchmark should get its own file to keep code flow as easy as possible to follow -- this is not about software engineering!
4. Update `CMakeLists.txt` if necessary.
5. Document the new benchmark and its command line switch in the `README.md` if suitable.

### Coding Style

The codebase follows modern C++20 guidelines. Use `-Wall -Wextra -Wpedantic`
for clean builds and keep functions small and well documented.

## About

Developed at the Chair for Computer Architecture and Parallel Systems
at the Technical University of Munich ([CAPS TUM](https://www.ce.cit.tum.de/en/caps/homepage/)).
Originally authored by Dominik Größler, completely reworked by Manuel Walter
Mußbacher and currently maintained by Stepan Vanecek. The research paper
surrounding this work can be found [here](https://doi.org/10.1145/3731599.3767518).

This project is licensed under the [Apache License 2.0](LICENSE).
