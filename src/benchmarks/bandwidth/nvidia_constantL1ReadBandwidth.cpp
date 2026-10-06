#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <tuple>

static constexpr auto WARMUP_REPS = 8;
static constexpr auto MS_PER_SECOND = 1000.0; // ms

__global__ void constantL1ReadBandwidthKernel(uint32_t* __restrict__ dst, size_t totalElements, size_t reps)
{
    const uint32_t tid = threadIdx.x;
    const size_t alignOffset = (reinterpret_cast<uintptr_t>(arr16384AscStride0) % sizeof(uint2)) / sizeof(uint32_t);
    const uint32_t* base = arr16384AscStride0 + alignOffset;

    uint32_t dummy = 0;

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = 0; i < totalElements; ++i)
        {
            uint64_t loaded = 0;

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile (
                "{\n\t"
                ".reg .u64 cp;\n\t"
                "cvta.to.const.u64 cp, %1;\n\t"
                "ld.const.u64 %0, [cp];\n\t"
                "}"
                : "=l"(loaded)
                : "l"(base + i * 2)
            );
            #endif

            dummy ^= static_cast<uint32_t>(loaded) ^ static_cast<uint32_t>(loaded >> 32);
        }
    }

    dst[tid] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> constantL1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    size_t totalElements = arraySizeBytes / sizeof(uint2);

    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);

    // Warm up
    constantL1ReadBandwidthKernel<<<1, numThreads, 0, stream>>>(d_dstArr, totalElements, WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    constantL1ReadBandwidthKernel<<<1, numThreads, 0, stream>>>(d_dstArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));

    // Bytes delivered to the threads, every thread receives the whole working set per rep.
    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) (totalElements * numThreads * sizeof(uint2)) * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark
{
    namespace nvidia
    {
        CacheBandwidthResult measureConstantL1ReadBandwidthSweep(size_t arraySizeBytes)
        {
            auto stream = util::createStreamForCU(0);

            uint32_t minNumThreads = util::getWarpSize();
            uint32_t maxNumThreads = util::getMaxThreadsPerBlock();
            size_t minReps = MIN_REPS;
            size_t maxReps = MAX_REPS;

            CacheBandwidthResult result{};
            result.measuredBandwidth = 0.0;
            result.dataBytes = arraySizeBytes;
            result.cycles = 0;
            result.time = 0.0;
            result.numThreads = 0;
            result.numBlocks = 1;
            result.numReps = 0;

            for (uint32_t numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads *= 2)
            {
                std::vector<double> bandwidthResults;

                result.threadsTested.push_back(numThreads);

                for (size_t reps = minReps; reps <= maxReps; reps *= 2)
                {
                    if (numThreads == minNumThreads)
                    {
                        result.repsTested.push_back(reps);
                    }

                    auto [timeS, bandwidth] = constantL1ReadBandwidthLauncher(arraySizeBytes, numThreads, reps, stream);

                    bandwidthResults.push_back(bandwidth);

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numReps = reps;
                    }
                }

                result.bandwidthGridGiBs.push_back(bandwidthResults);
            }

            util::hipCheck(hipStreamDestroy(stream));

            return result;
        }
    }
}
