#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR; // Factor
static constexpr auto MS_PER_SECOND = 1000.0; // ms
static constexpr auto WARMUP_REPS = 4; // warmup passes for the sweep (large working set)

// Mirrors the L2 bandwidth sweep: reads a working set that is much larger
// than the last level cache, so the traffic is served from main memory,
// while varying the block and thread count.
template <uint32_t LoadsPerBatch>
__global__ void mainMemoryReadBandwidthSweepKernel(uint32_t* __restrict__ dst, const uint32v4* __restrict__ src, size_t n, size_t reps)
{
    LOADS_PER_BATCH_CHECK(LoadsPerBatch, MAX_MULTI_BLOCK_LOADS_PER_BATCH);

    // read once, on AMD the "memory" clobber of the asm would reload them in every iteration
    const size_t blockStride = blockDim.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockStride;
    const size_t first = blockIdx.x * blockStride + threadIdx.x;

    uint32_t dummy = 0;

    for (size_t j = 0; j < reps; ++j)
    {
        dummy ^= gridReadPass<LoadsPerBatch, false>(src, first, n, gridStride);
    }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}

static std::tuple<double, double> mainMemoryReadBandwidthSweepLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps, util::AllocatorType allocType)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);
    const auto kernel = withLoadsPerBatch<MAX_MULTI_BLOCK_LOADS_PER_BATCH>(loadsPerBatch, [](auto n) { return mainMemoryReadBandwidthSweepKernel<decltype(n)::value>; });

    uint32v4* d_srcArr = util::allocateMemory<uint32v4>(numElements, allocType);
    uint32_t* d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);

    kernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, numElements, WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    kernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, numElements, reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::freeMemory(d_srcArr, allocType);
    util::hipCheck(hipFree(d_dstArr));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    CacheBandwidthResult measureMainMemoryReadBandwidthSweep(size_t mainMemorySizeBytes, util::AllocatorType allocType) {
        util::hipDeviceReset();

        // Main memory bandwidth must be measured with a SINGLE streaming pass over a
        // working set larger than the last level cache. Re-reading a bounded array
        // (the cache-benchmark MIN_REPS/MAX_REPS model) is served from L2 and grossly
        // over-reports bandwidth, so main memory uses one pass over a ~1 GiB set.
        size_t arraySizeBytes = util::min(mainMemorySizeBytes / SIZE_DOWN, static_cast<size_t>(1) * 1024 * 1024 * 1024);

        return sweepMultiBlockBandwidth(arraySizeBytes, {1}, [&](uint32_t numBlocks, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
            return mainMemoryReadBandwidthSweepLauncher(arraySizeBytes, numBlocks, numThreads, loadsPerBatch, reps, allocType);
        });
    }
}
