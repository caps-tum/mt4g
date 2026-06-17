#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <iostream>

static constexpr auto WARMUP_REPS = 512;

static constexpr auto MIN_REPS = 2;
static constexpr auto MAX_REPS = 262144; // 2 ^ 18

static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

static constexpr auto MS_PER_SECOND = 1000.0;// ms

__global__ void l2ReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n, size_t reps) 
{
    size_t tid;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0}; 

    for (size_t j = 0; j < reps; ++j) 
    {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;

        for (size_t i = tid; i < n; i += stride) 
        {
            uint32v4 loaded; 
            
            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(loaded.x) // int
                , "=r"(loaded.y) // int
                , "=r"(loaded.z) // int
                , "=r"(loaded.w) // int
                : "l"(src + i) // uint32v4*
            );
            #endif

            #ifdef __HIP_PLATFORM_AMD__
            const uint64_t addr = reinterpret_cast<uint64_t>(src + i);
            asm volatile(
                "flat_load_dwordx4 %0, %1 sc1\n" // bypass L1
                : "=v"(loaded) // uint32v4
                : "v"(addr) // uint32v4*
                : "memory"
            );
            #endif

            // XOR is efficient
            dummy.x ^= loaded.x;
        }
    }
    
    dst[threadIdx.x] = dummy; // prevent dead code elimination
}

static std::tuple<double, double> l2ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps) 
{ 
    // Initialize device Arrays
    // sizeof(uint32v4) = 16 bytes -> allows us to load 4 integers with one instruction -> probability 
    // of the bandwidth being limited by the memory bandwidth rather than compute is considerably higher
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(numThreads);

    // warm up L2
    l2ReadBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l2ReadBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4), reps);
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));

    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB); // Convert to GiB
    double timeS = elapsedMs / MS_PER_SECOND;
    
    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    double measureL2ReadBandwidth(size_t l2SizeBytes)
    {
        util::hipDeviceReset();

        const size_t arraySizeBytes = l2SizeBytes * 0.8; // 80% L2 size, L1 is bypassed
        uint32_t maxThreads = util::getDeviceProperties().maxThreadsPerBlock;
        uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;
        size_t maxReps = MAX_REPS / 4;

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) 
        {
            results[i] = std::get<1>(l2ReadBandwidthLauncher(arraySizeBytes, maxBlocks, maxThreads, maxReps));
        }

        return util::average(results);
    }

    CacheBandwidthResult measureL2ReadBandwidthSweep(size_t l2SizeBytes) 
    {
        util::hipDeviceReset();

        const size_t arraySizeBytes = l2SizeBytes * 0.8; // 80% L2 size, L1 is bypassed

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
                    
                    auto [timeS, bandwidth] = l2ReadBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps);
                    
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

        return result;
    }
}