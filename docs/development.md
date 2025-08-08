# Development Notes

This document provides an overview of the project's structure and guidance for contributors.

## Repository layout

```
Makefile     - Build configuration using hipcc
include/     - Public headers and utilities
src/         - Benchmark implementation and helpers
src/benchmarks/ - Platform specific benchmark code
src/utils/   - Common utilities such as command line parsing and file output
docs/        - Project documentation
results/     - Available sample results
```

## Adding a new benchmark

1. Implement the benchmark in `src/benchmarks/` and expose a suitable interface
   in `include/`.
2. Try to follow the pattern of `measureXXX()`, `XXXLauncher()` and `XXXKernel()` to keep the structure modular and readable.
   Every benchmark should get its own file to keep code flow as easy as possible to follow - this is not about software engineering!
4. Update `Makefile` if new source files or dependencies are required.
5. Document the new benchmark and its command line switch in `docs/usage.md`, if suitable.

## Coding style

The codebase follows modern C++20 guidelines. Use `-Wall -Wextra -Wpedantic` for clean builds and keep functions small and well documented.

## Building in debug mode

You can enable verbose intermediate files by setting `d=1` during compilation: This will drop all temporary files, especially (PTX/CDNA ISA), into ./dbg

```bash
make -j$(nproc) GPU_TARGET_ARCH=<sm_XX | gfxXXX> d=1
```

Temporary files will be placed in the `dbg/` directory for inspection.

