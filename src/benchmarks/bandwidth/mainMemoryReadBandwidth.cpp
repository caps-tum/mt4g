#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <string>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR;// Factor
static constexpr auto WARMUP_REPS = 4;// warmup passes for the sweep (large working set)

// Mirrors the L2 bandwidth sweep: re-reads a working set that is much larger
// than the last level cache `reps` times so the traffic is served from main
// memory, while varying the block and thread count.
template <uint32_t LoadsPerBatch>
__global__ void mainMemoryReadBandwidthSweepKernel(uint32_t* __restrict__ dst, const uint32v4* __restrict__ src, size_t n, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_MULTIPLE_BLOCKS);

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

// The kernel instance for a number of loads per batch
static auto mainMemoryReadBandwidthSweepKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:   return mainMemoryReadBandwidthSweepKernel<1>;
        case 2:   return mainMemoryReadBandwidthSweepKernel<2>;
        case 4:   return mainMemoryReadBandwidthSweepKernel<4>;
        case 8:   return mainMemoryReadBandwidthSweepKernel<8>;
        case 16:  return mainMemoryReadBandwidthSweepKernel<16>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


static std::tuple<double, double> mainMemoryReadBandwidthSweepLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps, util::AllocatorType allocType) {
    const auto kernel = mainMemoryReadBandwidthSweepKernelFor(loadsPerBatch);

    uint32v4* d_srcArr = util::allocateMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4), allocType);
    uint32_t* d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);

    kernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    kernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4), reps);
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

        CacheBandwidthResult result(arraySizeBytes, 0);
        initMultiBlockGrid(result, 1, 1); // single streaming pass (see note above)

        const size_t numBlockSteps = result.blocksTested.size();
        const size_t numThreadSteps = result.threadsTested.size();

        // Lowest thread count worth measuring; used as an index into threadsTested.
        // Once a thread sweep terminates, this and lower counts are skipped for lower blocks.
        size_t lowestThreadIndex = 0;

        // Search block counts and thread counts from highest to lowest.
        for (size_t bi = numBlockSteps; bi-- > 0; )
        {
            const uint32_t numBlocks = result.blocksTested[bi];
            double maxBandwidthThisBlock = 0.0;

            for (size_t ti = numThreadSteps; ti-- > lowestThreadIndex; )
            {
                const uint32_t numThreads = result.threadsTested[ti];

                double bestThisThread = 0.0;

                for (uint32_t loadsPerBatch = 1; loadsPerBatch <= MAX_LOADS_PER_BATCH_MULTIPLE_BLOCKS; loadsPerBatch *= 2)
                {
                    auto [timeS, bandwidth] = mainMemoryReadBandwidthSweepLauncher(arraySizeBytes, numBlocks, numThreads, loadsPerBatch, 1, allocType);

                    bestThisThread = std::max(bestThisThread, bandwidth);

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numBlocks = numBlocks;
                        result.numReps = 1;
                        result.loadsPerBatch = loadsPerBatch;
                    }
                }

                result.bandwidth3D[bi][ti][0] = bestThisThread;

                // A >=25% drop from the best BW ends this thread sweep
                // and skips this and lower thread counts for remaining blocks.
                if (maxBandwidthThisBlock > 0.0 &&
                    bestThisThread <= BANDWIDTH_EARLY_TERMINATION_FACTOR * maxBandwidthThisBlock)
                {
                    lowestThreadIndex = ti + 1;
                    break;
                }

                if (bestThisThread > maxBandwidthThisBlock)
                {
                    maxBandwidthThisBlock = bestThisThread;
                }
            }
        }

        return result;
    }
}