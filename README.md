# mt4g - Memory Topology 4 GPUs

**mt4g** is a vendor-agnostic collection of microbenchmarks and APIs that
explores the compute and memory topologies of both AMD and NVIDIA GPUs based on
the HIP toolchain. By capturing system properties such as the number of SMs/CUs,
warp size, memory and cache sizes, cache line sizes, load and store latencies
as well as exposing deep cache subsystems and their physical layouts, it
provides critical support for GPU performance modeling and analysis within one
unified interface.

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

The research paper of this work can be found [here](https://doi.org/10.1145/3731599.3767518).

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
export HIP_PATH=<path_to_spack_installation>/opt/spack/<system_architecture>/hip-<version>-<hash>
```

Additionally for NVIDIA targets, the `CUDA_PATH` environment variable needs to
be set to the CUDA installation directory.

**mt4g** has been tested successfully with `hip@6.3.3` and `cuda@12.8`.

### Build

Use the `GPU_TARGET_ARCH` build flag to select the target GPU architecture for
AMD (e.g. `gfx90a`) and NVIDIA (e.g. `sm_90`) respectively. To build and
install **mt4g**, run

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
| `-g, --graphs` | Generate graphs for each benchmark |
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
do not consider AMD's RDNA GPUs.

## Repository layout

```
include/          - Public headers and utilities
results/          - Available sample results
src/              - Benchmark implementation and CLI helpers
docs/             - Additional documentation
CMakeLists.txt    - Build configuration
```

See [docs/usage.md](docs/usage.md) for a comprehensive description of the
command line interface and [docs/development.md](docs/development.md) for
contribution guidelines.

## Sample results and contributions

Pre-measured results for selected GPUs live in the
[results](results/) directory. If your hardware is not yet listed,
we would greatly appreciate additional reports: run the tool with
`--raw --graphs --report` and open a pull request to share your measurements.

## Project background

Developed at the Chair for Computer Architecture and Parallel Systems (CAPS) at
the Technical University of Munich. Originally authored by Dominik Größler,
completely reworked by Manuel Walter Mußbacher and currently maintained by
Stepan Vanecek.

## Known issues and limitations

- L2 segment size measurements on AMD GPUs are currently unreliable due to the platform's complex cache behaviour.
- Constant L1.5 Cache Size detection is capped at 64 KiB. Denoted by 64 KiB + 1 and confidence = 0. (> 64 KiB)
- Bandwidths are not optimal because we currently do not use a (dynamically found) optimal number of blocks.
- Cache Line Size detection uses a heuristical approach and is therefore not guaranteed to be correct.
- Constant L1 shared with L1 is not too reliable. Hence, as a hotfix we repeat the measurements 10 times and on one unsuccessful run return not shared. We are working on a cleaner solution.
- Parallel build fails if depedencies were not fetched.
- Runs only on Linux.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

#### Benchmark Overview – NVIDIA

| Cache             | L1       | L2          | RO  | TXT | C1  | C1.5 | SM  | M    |
|-------------------|----------|-------------|-----|-----|-----|------|-----|------|
| **Size**          | Yes      | API, Seg.   | Yes | Yes | Yes | Yes  | API | API  |
| **Line Size**     | Yes      | Yes         | Yes | Yes | Yes | Yes  | –   | –    |
| **Fetch Gran.**   | Yes      | Yes         | Yes | Yes | Yes | Yes  | –   | –    |
| **Latency**       | Yes      | Yes         | Yes | Yes | Yes | Yes  | Yes | Yes  |
| **Count**         | Yes      | Yes, Seg.   | Yes | Yes | Yes | No   | –   | –    |
| **Miss Penalty**  | Yes      | Yes         | Yes | Yes | Yes | No   | –   | –    |
| **Bandwidth**     | No       | R/W         | No  | No  | No  | No   | No  | R/W  |
| **Shared With**   | RO, C1, TXT |      | L1, TXT | L1, RO    |     |      |     |      |

#### Benchmark Overview – AMD

| Cache             | vL1d     | L2          | L3  | sL1d | SM  | M    |
|-------------------|----------|-------------|-----|------|-----|------|
| **Size**          | Yes      | API, Seg.   | API | Yes  | API | API  |
| **Line Size**     | Yes      | API, FB     | API | Yes  | –   | –    |
| **Fetch Gran.**   | Yes      | Yes         | No  | Yes  | –   | –    |
| **Latency**       | Yes      | Yes         | No  | Yes  | Yes | Yes  |
| **Count**         | Yes      | API         | API | Uni. | –   | –    |
| **Miss Penalty**  | Yes      | Yes         | No  | Yes  | –   | –    |
| **Bandwidth**     | No       | R/W         | R/W | No   | No  | R/W  |
| **Shared With**   |          |             |     | CU   |     |      |


Seg. = Segment
Uni. = Unique
R/W = Read Bandwidth and Write Bandwidth
FB = Fallback Benchmark implemented
API = HIP Device Prop / HSA / AMDGPU KFD Kernel Module

