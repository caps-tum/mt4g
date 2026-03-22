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

__global__ void l1ReadBandwidthKernel(uint32v4* __restrict__ dst, const uint32v4* __restrict__ src, size_t totalElements, size_t reps) 
{
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint32v4 dummy {0, 0, 0, 0};

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            uint32v4 loaded;

            #ifdef __HIP_PLATFORM_AMD__
            const uint64_t addr = reinterpret_cast<uint64_t>(src + i);

            __asm__ volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr)
                : "memory"
            );
            #endif

            dummy.x ^= loaded.x;
        }
    }

    dst[gtid] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> l1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, hipStream_t stream) 
{
    const size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    const size_t totalThreads = static_cast<size_t>(numBlocks) * numThreads;

    // Allocate device arrays
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(totalElements);
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(totalThreads);

    // Warm up
    l1ReadBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    l1ReadBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark 
{
    namespace amd
    {
        CacheBandwidthResult measureL1ReadBandwidthBlockSweep(size_t arraySizeBytes) 
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

                        auto [timeS, bandwidth] = l1ReadBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, stream);
                        
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

