#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <limits>

static constexpr auto WARMUP_REPS = 512;


// Writes the array with all blocks; every rep starts at the data of the next block
__global__ void l3WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n, size_t reps)
{
    // read once, the "memory" clobber of the asm would reload them in every iteration
    const size_t blockStride = blockDim.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockStride;
    const uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockStride + threadIdx.x);
    const uint32v4 data = {tid, tid + 1, tid + 2, tid + 3};

    for (size_t j = 0; j < reps; ++j)
    {
        const size_t first = ((blockIdx.x + j) * blockStride + threadIdx.x) % gridStride;

        for (size_t i = first; i < n; i += gridStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile ("flat_store_dwordx4 %0, %1\n\t" : : "v"(dst + i), "v"(data) : "memory");
            #endif
        }
    }
}

static std::tuple<double, double> l3WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, util::AllocatorType allocType = util::AllocatorType::HipMalloc)
{
    uint32v4* d_dstArr = util::allocateMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4), allocType);

    // warm up
    l3WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l3WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));

    util::freeMemory(d_dstArr, allocType);

    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB); // Convert to GiB
    double timeS = elapsedMs / MS_PER_SECOND;
    
    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    namespace amd {
        CacheBandwidthResult measureL3WriteBandwidthSweep(size_t l2SizeBytes, size_t l3SizeBytes, util::AllocatorType allocType)
        {
            util::hipDeviceReset();

            size_t arraySizeBytes = util::max(l2SizeBytes * (util::getNumComputeDies() + 2), l3SizeBytes / 4);

            CacheBandwidthResult result(arraySizeBytes, 0);
            initMultiBlockGrid(result, MIN_REPS, MAX_REPS);

            const size_t numBlockSteps = result.blocksTested.size();
            const size_t numThreadSteps = result.threadsTested.size();
            const size_t numRepSteps = result.repsTested.size();

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

                    for (size_t ri = 0; ri < numRepSteps; ++ri)
                    {
                        const size_t reps = result.repsTested[ri];

                        auto [timeS, bandwidth] = l3WriteBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, allocType);

                        result.bandwidth3D[bi][ti][ri] = bandwidth;

                        if (bandwidth > bestThisThread)
                        {
                            bestThisThread = bandwidth;
                        }

                        if (bandwidth > result.measuredBandwidth)
                        {
                            result.measuredBandwidth = bandwidth;
                            result.time = timeS;
                            result.numThreads = numThreads;
                            result.numBlocks = numBlocks;
                            result.numReps = reps;
                        }
                    }

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
}
