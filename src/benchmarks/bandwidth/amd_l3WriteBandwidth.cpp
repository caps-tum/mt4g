#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <limits>

static constexpr auto WARMUP_REPS = 512;


static constexpr auto MS_PER_SECOND = 1000.0; // ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS; // rounds

__global__ void l3WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n, size_t reps) 
{
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    uint32v4 dummy = {tid, tid + 1, tid + 2, tid + 3};

    for (size_t j = 0; j < reps; ++j) 
    {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;

        for (size_t i = tid; i < n; i += stride) 
        {
            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_store_dwordx4 %0, %1\n"
                :
                : "v"(dst + i), "v"(dummy)
                : "memory"
            );
            #endif
        }
    }
}

static std::tuple<double, double> l3WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps) 
{
    uint32v4* d_dstArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));

    // warm up
    l3WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l3WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), reps);
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));

    util::hipCheck(hipFree(d_dstArr));

    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB); // Convert to GiB
    double timeS = elapsedMs / MS_PER_SECOND;
    
    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    namespace amd {
        double measureL3WriteBandwidth(size_t l2SizeBytes, size_t l3SizeBytes)
        {
            util::hipDeviceReset();

            const size_t arraySizeBytes = util::max(l2SizeBytes * (util::getNumXCDs() + 2), l3SizeBytes / 4);
            uint32_t maxThreads = util::getDeviceProperties().maxThreadsPerBlock;
            uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;
            size_t maxReps = MAX_REPS / 4;

            std::vector<double> results(ROUNDS);
            for (uint32_t i = 0; i < ROUNDS; ++i) 
            {
                results[i] = std::get<1>(l3WriteBandwidthLauncher(arraySizeBytes, maxBlocks, maxThreads, maxReps));
            }

            return util::average(results);
        }

        CacheBandwidthResult measureL3WriteBandwidthSweep(size_t l2SizeBytes, size_t l3SizeBytes) 
        {
            util::hipDeviceReset();

            size_t arraySizeBytes = util::max(l2SizeBytes * (util::getNumXCDs() + 2), l3SizeBytes / 4);

            uint32_t minThreads = util::getDeviceProperties().warpSize;
            uint32_t maxThreads = util::getDeviceProperties().maxThreadsPerBlock;

            uint32_t minBlocks = util::getNumberOfComputeUnits();
            uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

            size_t minReps = MIN_REPS;
            size_t maxReps = MAX_REPS;

            CacheBandwidthResult result{};
            result.measuredBandwidth = 0.0;
            result.dataBytes = arraySizeBytes;
            result.cycles = 0;
            result.time = 0.0;
            result.numThreads = 0;
            result.numBlocks = 0;
            result.numReps = 0;

            // Precompute full block/thread/rep axes for CSV alignment.
            // Sweep runs descending; axes remain ascending for unchanged grid layout.
            for (uint32_t numBlocks = minBlocks; numBlocks <= maxBlocks; numBlocks *= 2)
            {
                result.blocksTested.push_back(numBlocks);
            }
            for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
            {
                result.threadsTested.push_back(numThreads);
            }
            for (size_t reps = minReps; reps <= maxReps; reps *= 2)
            {
                result.repsTested.push_back(reps);
            }

            const size_t numBlockSteps = result.blocksTested.size();
            const size_t numThreadSteps = result.threadsTested.size();
            const size_t numRepSteps = result.repsTested.size();
            // NaN marks configurations skipped by early termination, distinguishing them
            // from genuine 0 GiB/s measurements.
            const double UNMEASURED = std::numeric_limits<double>::quiet_NaN();

            result.bandwidth3D.assign(numBlockSteps, std::vector<std::vector<double>>(
                numThreadSteps, std::vector<double>(numRepSteps, UNMEASURED)));

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

                        auto [timeS, bandwidth] = l3WriteBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps);

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
