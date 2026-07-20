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

__global__ void mainMemoryReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0}; 

    for (size_t i = tid; i < n; i += stride) {
        uint32v4 loaded; 
        
        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(loaded.x) // int
            , "=r"(loaded.y) // int
            , "=r"(loaded.z) // int
            , "=r"(loaded.w) // int
            : "l"(src + i) // uint32v4*
        );
        #endif

        #ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "flat_load_dwordx4 %0, %1\n" 
            : "=v"(loaded) // uint32v4
            : "s"(src + i) // uint32v4*
            :
        );
        #endif

        // XOR is efficient
        dummy.x ^= loaded.x;
    }

    dst[tid % blockDim.x] = dummy; // prevent dead code elimination
}

double mainMemoryReadBandwidthLauncher(size_t arraySizeBytes) { 
    util::hipDeviceReset(); 

    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU()); 
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    // Initialize device Arrays
    // sizeof(uint32v4) = 16 bytes -> allows us to load 4 integers with one instruction -> probability 
    // of the bandwidth being limited by the memory bandwidth rather than compute is considerably higher
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(maxThreadsPerBlock); // total threads
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    mainMemoryReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4));
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end);
}

// --- Optimal-configuration sweep -------------------------------------------
// Mirrors the L2 bandwidth sweep: re-reads a working set that is much larger
// than the last level cache `reps` times so the traffic is served from main
// memory, while varying the block and thread count.
__global__ void mainMemoryReadBandwidthSweepKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n, size_t reps) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0};

    for (size_t j = 0; j < reps; ++j) {
        for (size_t i = tid; i < n; i += stride) {
            uint32v4 loaded;

            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(loaded.x) // int
                , "=r"(loaded.y) // int
                , "=r"(loaded.z) // int
                , "=r"(loaded.w) // int
                : "l"(src + i) // uint32v4*
            );
            #endif

            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_load_dwordx4 %0, %1\n"
                : "=v"(loaded) // uint32v4
                : "s"(src + i) // uint32v4*
                :
            );
            #endif

            dummy.x ^= loaded.x;
        }
    }

    dst[tid % blockDim.x] = dummy; // prevent dead code elimination
}

static std::tuple<double, double> mainMemoryReadBandwidthSweepLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps) {
    uint32v4* d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4* d_dstArr = util::allocateGPUMemory<uint32v4>(numThreads);

    mainMemoryReadBandwidthSweepKernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    mainMemoryReadBandwidthSweepKernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4), reps);
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    double measureMainMemoryReadBandwidth(size_t mainMemorySizeBytes) {
        size_t testSizeBytes = mainMemorySizeBytes / SIZE_DOWN; // Divide by SIZE_DOWN to avoid too large memory allocations
        double testSizeGiB = (double)testSizeBytes / (double)(1 * GiB); // Convert to GiB

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) {
            results[i] = mainMemoryReadBandwidthLauncher(testSizeBytes) / MS_PER_SECOND;
        }

        return testSizeGiB / util::average(results);
    }

    CacheBandwidthResult measureMainMemoryReadBandwidthSweep(size_t mainMemorySizeBytes) {
        util::hipDeviceReset();

        // Main memory bandwidth must be measured with a SINGLE streaming pass over a
        // working set larger than the last level cache. Re-reading a bounded array
        // (the cache-benchmark MIN_REPS/MAX_REPS model) is served from L2 and grossly
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

                auto [timeS, bandwidth] = mainMemoryReadBandwidthSweepLauncher(arraySizeBytes, numBlocks, numThreads, 1);

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