#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <numeric>

static constexpr auto WARMUP_REPS = 128;


// Each wave writes every numWaves-th 64-byte element with four scalar stores
__global__ void sL1WriteBandwidthKernel([[maybe_unused]] uint32v16* __restrict__ dst, uint64_t* __restrict__ timingResult, uint32_t waveSize, uint32_t numElements, size_t reps)
{
    const uint32_t numWaves = blockDim.x / waveSize;
    uint32_t waveId = threadIdx.x / waveSize;

    #ifdef __HIP_PLATFORM_AMD__
    // unlike inline asm, the builtin marks the result wave-uniform, so data and addresses stay in SGPRs as s_store requires
    waveId = __builtin_amdgcn_readfirstlane(waveId);
    #endif

    [[maybe_unused]] const uint32v4 data {waveId, waveId + 1, waveId + 2, waveId + 3};

    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (uint32_t i = waveId; i < numElements; i += numWaves)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "s_store_dwordx4 %0, %1, 0\n\t"
                "s_store_dwordx4 %0, %1, 16\n\t"
                "s_store_dwordx4 %0, %1, 32\n\t"
                "s_store_dwordx4 %0, %1, 48\n\t"
                :
                : "s"(data), "s"(dst + i)
                : "memory"
            );
            #endif
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile ("s_waitcnt lgkmcnt(0)\n\t" : : : "memory"); // the compiler does not track the asm stores
    #endif

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sL1WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps)
{
    const uint32_t numElements = arraySizeBytes / sizeof(uint32v16);

    uint32v16 *d_dstArr = util::allocateGPUMemory<uint32v16>(numElements);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    double timeS = 0.0;

    // warm up; a block size that needs more registers than available does not launch
    sL1WriteBandwidthKernel<<<1, numThreads>>>(d_dstArr, d_timingResult, util::getWarpSize(), numElements, WARMUP_REPS);
    const hipError_t launchError = hipGetLastError();

    if (launchError != hipErrorLaunchOutOfResources)
    {
        util::hipCheck(launchError);

        sL1WriteBandwidthKernel<<<1, numThreads>>>(d_dstArr, d_timingResult, util::getWarpSize(), numElements, reps);
        util::hipCheck(hipGetLastError());
        util::hipCheck(hipStreamSynchronize(0));
        timeS = globalTimerSeconds(d_timingResult);
    }

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) numElements * sizeof(uint32v16) * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}

namespace benchmark {
    namespace amd {
        CacheBandwidthResult measureScalarL1WriteBandwidthSweep(size_t arraySizeBytes) 
        {            
            uint32_t minNumThreads = util::getWarpSize();
            uint32_t maxNumThreads = util::getMaxThreadsPerBlock();
            size_t minReps = MIN_REPS;
            size_t maxReps = MAX_REPS;

            CacheBandwidthResult result(arraySizeBytes, 1);

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

                    auto [timeS, bandwidth] = sL1WriteBandwidthLauncher(arraySizeBytes, numThreads, reps);
                    
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

            return result;
        }
    }
}