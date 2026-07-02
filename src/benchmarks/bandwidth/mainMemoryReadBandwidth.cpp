#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR;// Factor
static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void mainMemoryReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0};

    for (size_t i = tid; i < n; i += stride) {
        uint32v4 loaded;

        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
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
                "global_load_dwordx4 %0, %1, off " GLC_SLC "\n\t"
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

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}

double mainMemoryReadBandwidthLauncher(size_t arraySizeBytes) {
    util::hipDeviceReset();

    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(maxThreadsPerBlock);

    size_t n = arraySizeBytes / sizeof(uint32v4);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipEventRecord(start));
    mainMemoryReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, n);
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end);
}

namespace benchmark {
    double measureMainMemoryReadBandwidth(size_t mainMemorySizeBytes) {

        size_t testSizeBytes = std::min(mainMemorySizeBytes / SIZE_DOWN, static_cast<size_t>(16 * GiB));
        size_t largestCacheBytes = util::getL3SizeBytes().value_or(
                                    util::getL2SizeBytes().value_or(
                                     util::getL1SizeBytes().value_or(0)));
        if (largestCacheBytes > 0)
            testSizeBytes = std::max(testSizeBytes, largestCacheBytes * 4);
        double testSizeGiB = (double)testSizeBytes / (double)(1 * GiB); // Convert to GiB

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) {
            results[i] = mainMemoryReadBandwidthLauncher(testSizeBytes) / MS_PER_SECOND;
        }

        return testSizeGiB / util::average(results);
    }
}
