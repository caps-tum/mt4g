#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cstdlib>
#include <string>

static constexpr auto WARMUP_REPS = 8;


static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

// Texture cache read bandwidth benchmark for a single SM. Mirrors the L1 benchmark 
// but measures the texture fetch path (tex1Dfetch) instead of normal global loads.
__global__ void textureReadBandwidthKernel([[maybe_unused]] hipTextureObject_t tex, uint32_t* __restrict__ dst, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps)
{
    const uint32_t tid = threadIdx.x;
    const size_t base = static_cast<size_t>(tid) * elementsPerThread;

    uint32_t dummy = 0;

    // Warm up the texture / unified cache
    for (size_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            #ifdef __HIP_PLATFORM_NVIDIA__
            int4 loaded = tex1Dfetch<int4>(tex, static_cast<int>(base + i));
            dummy ^= static_cast<uint32_t>(loaded.x);
            #endif
        }
    }

    uint64_t start = 0, end = 0;

    __syncthreads();

    if (tid == 0)
    {
        #ifdef __HIP_PLATFORM_NVIDIA__
        __asm__ volatile (
            "mov.u64 %0, %%clock64;\n\t"
            : "=l"(start)
            :
            : "memory"
        );
        #endif
    }

    __syncthreads();

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            #ifdef __HIP_PLATFORM_NVIDIA__
            int4 loaded = tex1Dfetch<int4>(tex, static_cast<int>(base + i));
            dummy ^= static_cast<uint32_t>(loaded.x);
            #endif
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        #ifdef __HIP_PLATFORM_NVIDIA__
        __asm__ volatile (
            "mov.u64 %0, %%clock64;\n\t"
            : "=l"(end)
            :
            : "memory"
        );
        #endif

        *timing_result = end - start;
    }

    dst[tid] = dummy; // prevent dead code elimination
}


static std::tuple<uint64_t, double, double> textureReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps)
{
    size_t totalElements = arraySizeBytes / sizeof(int4);
    size_t elementsPerThread = totalElements / numThreads;

    int4 *d_srcArr = util::allocateGPUMemory<int4>(totalElements);
    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    hipTextureObject_t tex = util::createTextureObject<int4>(d_srcArr, totalElements);

    // Run the kernel
    textureReadBandwidthKernel<<<1, numThreads>>>(tex, d_dstArr, d_timingResult, elementsPerThread, reps);

    // Get the timings from the device
    std::vector<uint64_t> timingResult = util::copyFromDevice<uint64_t>(d_timingResult, 1);

    util::hipCheck(hipDestroyTextureObject(tex));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    // calculate the bandwidth
    double gpuClockHz = util::getClockRateKHz() * 1000.0;
    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    double timeS = (double) timingResult[0] / gpuClockHz;

    // return (cycles, time in seconds, measured bandwidth)
    return {timingResult[0], timeS, dataGiB / timeS};
}


namespace benchmark
{
    namespace nvidia
    {
        double measureTextureReadBandwidth(size_t arraySizeBytes)
        {
            std::vector<double> results(ROUNDS);

            uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;
            size_t maxReps = MAX_REPS;

            for (uint32_t i = 0; i < ROUNDS; ++i)
            {
                results[i] = std::get<2>(textureReadBandwidthLauncher(arraySizeBytes, maxNumThreads, maxReps));
            }

            return util::average(results);
        }

        CacheBandwidthResult measureTextureReadBandwidthSweep(size_t arraySizeBytes)
        {
            uint32_t minNumThreads = util::getDeviceProperties().warpSize;
            uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;
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

                    auto [cycles, timeS, bandwidth] = textureReadBandwidthLauncher(arraySizeBytes, numThreads, reps);

                    bandwidthResults.push_back(bandwidth);

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.cycles = cycles;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numReps = reps;
                    }
                }

                result.bandwidthGridGiBs.push_back(bandwidthResults);
            }

            return result;
        }
    }
}
