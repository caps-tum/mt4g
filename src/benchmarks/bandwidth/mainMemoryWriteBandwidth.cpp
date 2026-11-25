#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>
#include <tuple>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR; // Factor
static constexpr auto WARMUP_REPS = 4; // warmup passes for the sweep (large working set)

// Mirrors the L2 bandwidth sweep: writes a working set that is much larger than
// the last level cache `reps` times so the traffic reaches main memory, while
// varying the block and thread count.
__global__ void mainMemoryWriteBandwidthSweepKernel(uint32v4* __restrict__ dst, size_t n, size_t reps)
{
    // read once, on AMD the "memory" clobber of the asm would reload them in every iteration
    const size_t blockStride = blockDim.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockStride;
    const size_t first = blockIdx.x * blockStride + threadIdx.x;
    const uint32_t tid = static_cast<uint32_t>(first);
    const uint32v4 data = {tid, tid + 1, tid + 2, tid + 3};

    for (size_t j = 0; j < reps; ++j)
    {
        for (size_t i = first; i < n; i += gridStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile ("flat_store_dwordx4 %0, %1\n\t" : : "v"(dst + i), "v"(data) : "memory");
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile ("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};\n\t" : : "l"(dst + i), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w) : "memory");
            #endif
        }
    }
}

static std::tuple<double, double> mainMemoryWriteBandwidthSweepLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, util::AllocatorType allocType)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);

    uint32v4 *d_dstArr = util::allocateMemory<uint32v4>(numElements, allocType);

    mainMemoryWriteBandwidthSweepKernel<<<numBlocks, numThreads>>>(d_dstArr, numElements, WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    mainMemoryWriteBandwidthSweepKernel<<<numBlocks, numThreads>>>(d_dstArr, numElements, reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::freeMemory(d_dstArr, allocType);

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

    return {timeS, dataGiB / timeS};
}


namespace benchmark {
    CacheBandwidthResult measureMainMemoryWriteBandwidthSweep(size_t mainMemorySizeBytes, util::AllocatorType allocType) {
        util::hipDeviceReset();

        // Main memory bandwidth must be measured with a SINGLE streaming pass over a
        // working set larger than the last level cache. Re-writing a bounded array
        // (the cache-benchmark MIN_REPS/MAX_REPS model) is absorbed by L2 and grossly
        // over-reports bandwidth, so main memory uses one pass over a ~1 GiB set.
        size_t arraySizeBytes = util::min(mainMemorySizeBytes / SIZE_DOWN, static_cast<size_t>(1) * 1024 * 1024 * 1024);

        return sweepMultiBlockBandwidth(arraySizeBytes, {1}, [&](uint32_t numBlocks, uint32_t numThreads, uint32_t, size_t reps) {
            return mainMemoryWriteBandwidthSweepLauncher(arraySizeBytes, numBlocks, numThreads, reps, allocType);
        }, 1);
    }
}
