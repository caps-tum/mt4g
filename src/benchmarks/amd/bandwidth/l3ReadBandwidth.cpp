#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>

static constexpr auto MS_PER_SECOND = 1000.0; // ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS; // rounds

__global__ void l3ReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n) {
    size_t tid;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0};

    for (size_t j = 0; j < blockDim.x; ++j) {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;

        for (size_t i = tid; i < n; i += stride) {
            uint32v4 loaded;
            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_load_dwordx4 %0, %1 " GLC_SLC "\n"
                : "=v"(loaded)
                : "v"(src + i)
                : "memory"
            );
            #endif
            dummy.x ^= loaded.x;
        }
    }

    tid = blockIdx.x * blockDim.x + threadIdx.x;
    dst[tid] = dummy; // prevent dead code elimination
}

double l3ReadBandwidthLauncher(size_t arraySizeBytes) {
    util::hipCheck(hipDeviceReset());

    uint32_t maxThreadsPerBlock = util::getMaxThreadsPerBlock();
    uint32_t maxBlocks = util::getNumberOfComputeUnits();

    uint32v4* d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4* d_dstArr = util::allocateGPUMemory<uint32v4>(maxBlocks * maxThreadsPerBlock);

    l3ReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4));

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l3ReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4));
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end) / maxThreadsPerBlock;
}

namespace benchmark {
    namespace amd {
        double measureL3ReadBandwidth(size_t l3SizeBytes) {
            double testSizeGiB = static_cast<double>(l3SizeBytes) / (1 * GiB);

            std::vector<double> results(ROUNDS);
            for (uint32_t i = 0; i < ROUNDS; ++i) {
                results[i] = l3ReadBandwidthLauncher(l3SizeBytes) / MS_PER_SECOND;
            }

            return testSizeGiB / util::average(results);
        }
    }
}
