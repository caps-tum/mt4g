# mt4g - Memory Topology 4 GPUs

**mt4g** is a HIP‑based collection of microbenchmarks that explores the memory
hierarchy of modern GPUs. It measures cache sizes, line sizes, latencies,
resource‑sharing behaviour, ... on both NVIDIA and AMD hardware and emits automatically 
evaluated results as structured JSON.

## Available sample results

| GPU | Report |
| --- | ------ |
| NVIDIA P6000  | [Details](results/Quadro%20P6000/README.md) |
| NVIDIA V100   | [Details](results/Tesla%20V100-PCIE-16GB/README.md) |
| NVIDIA H100   | [Details](results/NVIDIA%20H100%2080GB%20HBM3/README.md) |
| AMD MI100     | [Details](results/AMD%20Instinct%20MI100/README.md) |
| AMD MI210     | [Details](results/AMD%20Instinct%20MI210/README.md) |
| AMD MI300X VF | [Details](results/AMD%20Instinct%20MI300X%20VF/README.md) |

## Features

- Unified build system for NVIDIA (`sm_XX`) and AMD (`gfxXXXX`) targets
- Benchmarks for L1/L2/L3 caches, scalar caches, shared and main memory
- Optional NVIDIA‑specific constant, read‑only and texture cache tests
- Graph generation and raw timing export
- JSON output summarising all measured metrics

## Requirements

    - HIP SDK with the `hipcc` compiler
    - GPU drivers and runtime libraries
    - `HIP_PATH` environment variable pointing to the HIP installation
    - `GPU_TARGET_ARCH` set to the desired architecture (e.g. `sm_70`, `gfx90a`)
    - Python 3 with the `matplotlib`, `pandas` and `numpy` packages for graph generation

The project has been verified with CUDA 12.8 and `hipcc` 6.3.3.

## Build

A suitable HIP environment can be obtained most easily via
[Spack](https://spack.io):

```bash
spack install hip           # for AMD targets
spack install hip cuda      # includes NVCC backend for NVIDIA targets
spack load hip              # sets HIP_PATH and exposes hipcc
```

Make sure to set `HIP_PATH` and `CUDA_PATH` when compiling for NVIDIA.
Choose the desired GPU architecture and invoke the build:
Note that you may have to run make twice if it fails because of missing dependencies.

```bash
make -j$(nproc) GPU_TARGET_ARCH=<sm_XX|gfxXXX>
```

External dependencies (`cxxopts`, `nlohmann/json`) are fetched automatically
when missing.

## Usage

```bash
./mt4g [options]
```

Common options:

| Option | Description |
| ------ | ----------- |
| `-d, --device-id <id>` | GPU device to use (default `0`) |
| `-g, --graphs` | Generate graphs for each benchmark |
| `-o, --raw` | Write raw timing data |
| `-p, --report` | Create Markdown report in output directory |
| `-j, --json` | Save final JSON output to `<GPU_NAME>.json` in the current directory |
| `-r, --random` | Randomize P-Chase arrays |
| `-q, --quiet` | Reduce console output |
| `--l1`, `--l2`, `--l3` | Run cache benchmarks for selected levels |
| `--scalar`, `--shared`, `--memory` | Run scalar, shared and main memory tests |
| `--constant`, `--readonly`, `--texture` | NVIDIA specific cache benchmarks |
| `--resourceshare` | Run resource sharing benchmarks |
| `-h, --help` | Show full help |

If no benchmark group is chosen all available groups are executed. Unsupported
groups are disabled automatically depending on the platform.

Make sure to have exclusive GPU access, otherwise results are far less reliable.

### Output

Benchmark results are printed as JSON. With `-j`/`--json` the final output is
additionally saved as `<GPU_NAME>.json` in the current working directory. When
graph, raw or report output is enabled the files are written to a directory
named after the detected GPU. The `--report` flag writes a `README.md`
containing the JSON summary and embeds all generated graphs with links to raw
data.

## Repository layout

```
include/   - Public headers and utilities
results/   - Available sample results
src/       - Benchmark implementation and CLI helpers
docs/      - Additional documentation
Makefile   - Build configuration
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
