# Usage Guide

This document explains how to build and run **mt4g** and describes all command line options.

## Building

1. Install a HIP environment. The simplest approach is using
   [Spack](https://spack.io):

   ```bash
   spack install hip           # for AMD targets
   spack install hip cuda      # includes NVCC backend for NVIDIA
   spack load hip              # exposes hipcc and sets HIP_PATH
   ```

2. Ensure the `HIP_PATH` environment variable points to the HIP installation (Spack should handle this automatically). 
3. Choose the target GPU architecture (e.g. `sm_70` for NVIDIA or `gfx90a` for
   AMD) and pass it as `GPU_TARGET_ARCH`.
4. Build the project:

   ```bash
   make -j$(nproc) GPU_TARGET_ARCH=sm_70
   ```

The first build automatically fetches the required third‑party libraries
`cxxopts` and `nlohmann/json` into `external/`. The tool has been tested with
CUDA 12.8 and `hipcc` 6.3.3.

## Running

Invoke the binary and select which benchmark groups should be executed:

e.g. 
```bash
./mt4g --l1 --l2 --memory
```

If no benchmark group is selected all available groups are executed by default. Platform‑specific groups are disabled automatically when not supported.

### Command line options

| Option | Description |
| ------ | ----------- |
| `-d, --device-id <id>` | GPU device to use (default `0`) |
| `-g, --graphs` | Generate graphs for each benchmark |
| `-o, --raw` | Write raw timing data |
| `-p, --report` | Create Markdown report in output directory |
| `-r, --random` | Randomize P-Chase arrays |
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
| `-h, --help` | Display a detailed help message and exit |

Generating graphs requires Python 3 (python3 in PATH) with the `matplotlib`, `pandas` and `numpy` packages.

### Output

Benchmark results are printed as JSON to `stdout`. When `--graphs`, `--raw` or
`--report` is enabled, additional files are written to `./results/<GPU_NAME>`. 
The `--report` flag generates a `README.md` that embeds all
graphs and links to the raw data. If you would like to contribute results for
hardware not yet covered, please run the tool with `--raw --graphs --report` and send us
the generated directory via pull request.

