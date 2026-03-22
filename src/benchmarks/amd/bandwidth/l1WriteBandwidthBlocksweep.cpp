#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>

static constexpr auto WARMUP_REPS = 128;

static constexpr auto MIN_REPS = 2;
static constexpr auto MAX_REPS = 262144; // 2 ^ 18

static constexpr auto MS_PER_SECOND = 1000.0; // ms

__global__ void l1WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t totalElements, size_t reps)
{
    const uint32_t gtid = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;

    const uint32v4 dummy = {gtid, gtid + 1, gtid + 2, gtid + 3};

    #ifdef __HIP_PLATFORM_AMD__
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(dst);
    #endif

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "flat_store_dwordx4 %0, %1\n\t"
                :
                : "v"(baseAddr + i * sizeof(uint32v4)),
                  "v"(dummy)
                : "memory"
            );
            #endif
        }
    }
}

static std::tuple<double, double> l1WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    const size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    const size_t totalThreads = static_cast<size_t>(numBlocks) * numThreads;

    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(totalThreads);
    
    //warm up
    l1WriteBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, totalElements, WARMUP_REPS);
    
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    l1WriteBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark {
    namespace amd
    {
        CacheBandwidthResult measureL1WriteBandwidthBlockSweep(size_t arraySizeBytes) 
        {
            // pin the entire sweep to a single CU.
            auto stream = util::createStreamForCU(0);

            uint32_t minThreads = util::getDeviceProperties().warpSize;
            uint32_t maxThreads = util::getDeviceProperties().maxThreadsPerBlock;
            uint32_t minBlocks = 1;
            uint32_t maxBlocks = util::getDeviceProperties().maxBlocksPerMultiProcessor;
            
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

            for (uint32_t numBlocks = minBlocks; numBlocks <= maxBlocks; numBlocks *= 2)
            {
                std::vector<std::vector<double>> threadsResults;

                result.blocksTested.push_back(numBlocks);

                for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
                {
                    std::vector<double> repsResults;

                    if (numBlocks == minBlocks)
                    {
                        result.threadsTested.push_back(numThreads);
                    }

                    for (size_t reps = minReps; reps <= maxReps; reps *= 2)
                    {
                        if (numBlocks == minBlocks && numThreads == minThreads)
                        {
                            result.repsTested.push_back(reps);
                        }

                        auto [timeS, bandwidth] = l1WriteBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, stream);

                        repsResults.push_back(bandwidth);

                        if (bandwidth > result.measuredBandwidth)
                        {
                            result.measuredBandwidth = bandwidth;
                            result.time = timeS;
                            result.numThreads = numThreads;
                            result.numBlocks = numBlocks;
                            result.numReps = reps;
                        }
                    }

                    threadsResults.push_back(repsResults);
                }

                result.bandwidth3D.push_back(threadsResults);
            }

            util::hipCheck(hipStreamDestroy(stream));

            return result;
        }
    }
}
