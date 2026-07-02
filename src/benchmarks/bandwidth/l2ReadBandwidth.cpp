#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>


static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void l2ReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n) {
    size_t tid;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0};

    for (size_t j = 0; j < blockDim.x; ++j) {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;

        for (size_t i = tid; i < n; i += stride) {
            uint32v4 loaded;

            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(loaded.x) // int
                , "=r"(loaded.y) // int
                , "=r"(loaded.z) // int
                , "=r"(loaded.w) // int
                : "l"(src + i) // uint32v4*
            );
            #endif

            #ifdef __HIP_PLATFORM_AMD__
            {
                uint64_t __addr = reinterpret_cast<uint64_t>(src + i);
                asm volatile(
                    "global_load_dwordx4 %0, %1, off\n\t"
                    "s_waitcnt vmcnt(0)\n\t"
                    : "=v"(loaded)
                    : "v"(__addr)
                    : "memory"
                );
            }
            #endif

            // XOR is efficient
            dummy.x ^= loaded.x;
        }
    }

    tid = blockIdx.x * blockDim.x + threadIdx.x;
    dst[tid % blockDim.x] = dummy; // prevent dead code elimination
}

double l2ReadBandwidthLauncher(size_t arraySizeBytes) {
    util::hipDeviceReset();

    // Calculate number of blocks and threads
    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    // Initialize device Arrays
    // sizeof(uint32v4) = 16 bytes -> allows us to load 4 integers with one instruction -> probability
    // of the bandwidth being limited by the memory bandwidth rather than compute is considerably higher
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(maxThreadsPerBlock); // total threads

    size_t n = arraySizeBytes / sizeof(uint32v4);

    // warm up L2
    l2ReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, n);
    util::hipCheck(hipDeviceSynchronize());

    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipEventRecord(start));
    l2ReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, n);
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end) / maxThreadsPerBlock; // Diff between end and start is blockDim.x * TimeItTakesToLoadArraySizeBytes
}

namespace benchmark {
    double measureL2ReadBandwidth(size_t l2SizeBytes) {
        size_t arraySizeBytes = l2SizeBytes * 0.9;
        double testSizeGiB = (double)arraySizeBytes / (double)(1 * GiB); // Convert to GiB

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) {
            results[i] = l2ReadBandwidthLauncher(arraySizeBytes) / MS_PER_SECOND;
        }

        return testSizeGiB / util::average(results);
    }
}
