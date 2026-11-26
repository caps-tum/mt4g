#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string>

static constexpr auto WARMUP_REPS = 512;


// Reads the array with all blocks; every rep starts at the data of the next block
template <uint32_t LoadsPerBatch>
__global__ void l3ReadBandwidthKernel(uint32_t* __restrict__ dst, const uint32v4* __restrict__ src, size_t n, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_MULTIPLE_BLOCKS);

    // read once, the "memory" clobber of the asm would reload them in every iteration
    const size_t blockStride = blockDim.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockStride;

    uint32_t dummy = 0;

    for (size_t j = 0; j < reps; ++j)
    {
        const size_t first = ((blockIdx.x + j) * blockStride + threadIdx.x) % gridStride;
        dummy ^= gridReadPass<LoadsPerBatch, false>(src, first, n, gridStride);
    }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}

// The kernel instance for a number of loads per batch
static auto l3ReadBandwidthKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:   return l3ReadBandwidthKernel<1>;
        case 2:   return l3ReadBandwidthKernel<2>;
        case 4:   return l3ReadBandwidthKernel<4>;
        case 8:   return l3ReadBandwidthKernel<8>;
        case 16:  return l3ReadBandwidthKernel<16>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


static std::tuple<double, double> l3ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps, util::AllocatorType allocType = util::AllocatorType::HipMalloc)
{
    const auto kernel = l3ReadBandwidthKernelFor(loadsPerBatch);

    uint32v4* d_srcArr = util::allocateMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4), allocType);
    uint32_t* d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);

    // warm up
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

    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB); // Convert to GiB
    double timeS = elapsedMs / MS_PER_SECOND;
    
    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    namespace amd {
        CacheBandwidthResult measureL3ReadBandwidthSweep(size_t l2SizeBytes, size_t l3SizeBytes, util::AllocatorType allocType)
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

                        double bestThisCell = 0.0;

                        for (uint32_t loadsPerBatch = 1; loadsPerBatch <= MAX_LOADS_PER_BATCH_MULTIPLE_BLOCKS; loadsPerBatch *= 2)
                        {
                            auto [timeS, bandwidth] = l3ReadBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, loadsPerBatch, reps, allocType);

                            if (bandwidth > bestThisCell)
                            {
                                bestThisCell = bandwidth;
                            }

                            if (bandwidth > result.measuredBandwidth)
                            {
                                result.measuredBandwidth = bandwidth;
                                result.time = timeS;
                                result.numThreads = numThreads;
                                result.numBlocks = numBlocks;
                                result.numReps = reps;
                                result.loadsPerBatch = loadsPerBatch;
                            }
                        }

                        result.bandwidth3D[bi][ti][ri] = bestThisCell;

                        if (bestThisCell > bestThisThread)
                        {
                            bestThisThread = bestThisCell;
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
