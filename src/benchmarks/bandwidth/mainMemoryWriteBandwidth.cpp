#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR;// Factor
static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds
static constexpr auto WARMUP_REPS = 4;// warmup passes for the sweep (large working set)

__global__ void mainMemoryWriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n) {
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    uint32v4 dummy = { tid, tid + 1, tid + 2, tid + 3 }; 

    for (size_t i = tid; i < n; i += stride) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(dst + i) // uint32v4*
            , "r"(dummy.x) // int
            , "r"(dummy.y) // int
            , "r"(dummy.z) // int
            , "r"(dummy.w) // int
        );
        #endif

        #ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "flat_store_dwordx4 %0, %1\n"
            :
            : "s"(dst + i) // uint32v4*
            , "v"(dummy) // uint32v4
            : "memory"
        );
        #endif
    }
}

double mainMemoryWriteBandwidthLauncher(size_t arraySizeBytes) { 
    util::hipDeviceReset(); 

    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU()); 
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    mainMemoryWriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4));
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end);
}

// --- Optimal-configuration sweep -------------------------------------------
// Mirrors the L2 bandwidth sweep: writes a working set that is much larger than
// the last level cache `reps` times so the traffic reaches main memory, while
// varying the block and thread count.
__global__ void mainMemoryWriteBandwidthSweepKernel(uint32v4* __restrict__ dst, size_t n, size_t reps) {
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    uint32v4 dummy = { tid, tid + 1, tid + 2, tid + 3 };

    for (size_t j = 0; j < reps; ++j) {
        for (size_t i = tid; i < n; i += stride) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};\n"
                :
                : "l"(dst + i) // uint32v4*
                , "r"(dummy.x) // int
                , "r"(dummy.y) // int
                , "r"(dummy.z) // int
                , "r"(dummy.w) // int
            );
            #endif

            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_store_dwordx4 %0, %1\n"
                :
                : "s"(dst + i) // uint32v4*
                , "v"(dummy) // uint32v4
                : "memory"
            );
            #endif
        }
    }
}

static std::tuple<double, double> mainMemoryWriteBandwidthSweepLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps) {
    uint32v4* d_dstArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));

    mainMemoryWriteBandwidthSweepKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    mainMemoryWriteBandwidthSweepKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), reps);
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    double measureMainMemoryWriteBandwidth(size_t mainMemorySizeBytes) {
        size_t testSizeBytes = mainMemorySizeBytes / SIZE_DOWN; // Divide by SIZE_DOWN to avoid too large memory allocations
        double testSizeGiB = (double)testSizeBytes / (double)(1 * GiB); // Convert to GiB

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) {
            results[i] = mainMemoryWriteBandwidthLauncher(testSizeBytes) / MS_PER_SECOND;
        }

        return testSizeGiB / util::average(results);
    }

    CacheBandwidthResult measureMainMemoryWriteBandwidthSweep(size_t mainMemorySizeBytes) {
        util::hipDeviceReset();

        // Main memory bandwidth must be measured with a SINGLE streaming pass over a
        // working set larger than the last level cache. Re-writing a bounded array
        // (the cache-benchmark MIN_REPS/MAX_REPS model) is absorbed by L2 and grossly
        // over-reports bandwidth, so main memory uses one pass over a ~1 GiB set.
        size_t arraySizeBytes = util::min(mainMemorySizeBytes / SIZE_DOWN, static_cast<size_t>(1) * 1024 * 1024 * 1024);

        uint32_t minThreads = util::getDeviceProperties().warpSize;
        uint32_t maxThreads = util::getDeviceProperties().maxThreadsPerBlock;

        uint32_t minBlocks = util::getNumberOfComputeUnits();
        uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

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

                if (numBlocks == minBlocks && numThreads == minThreads)
                {
                    result.repsTested.push_back(1); // single streaming pass (see note above)
                }

                auto [timeS, bandwidth] = mainMemoryWriteBandwidthSweepLauncher(arraySizeBytes, numBlocks, numThreads, 1);

                repsResults.push_back(bandwidth);

                if (bandwidth > result.measuredBandwidth)
                {
                    result.measuredBandwidth = bandwidth;
                    result.time = timeS;
                    result.numThreads = numThreads;
                    result.numBlocks = numBlocks;
                    result.numReps = 1;
                }

                threadsResults.push_back(repsResults);
            }

            result.bandwidth3D.push_back(threadsResults);
        }

        return result;
    }
}