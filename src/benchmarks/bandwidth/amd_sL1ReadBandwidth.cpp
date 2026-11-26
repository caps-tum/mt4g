#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>

static constexpr auto WARMUP_REPS = 128;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0 && WARMUP_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0, "loads per batch must divide the repetitions");


#ifdef __HIP_PLATFORM_AMD__
#define AMD_SCALAR_LOAD(offset) "s_load_dwordx16 s[40:55], %0, " #offset "\n\t"

// All loads write s40-s55 (the data is not used) and the wait is in the same asm,
// so the compiler cannot reuse these registers while the loads are in flight
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ void amdScalarLoadBatch(const uint32v16 *src)
{
    AMD_SINGLE_BLOCK_BATCH(LoadsPerBatch, AMD_SCALAR_LOAD, "s_waitcnt lgkmcnt(0)\n\t", "s"(src),
        "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52", "s53", "s54", "s55")
}
#endif


// Each wave reads every numWaves-th 64-byte element with scalar loads
template <uint32_t LoadsPerBatch>
__global__ void sL1ReadBandwidthKernel([[maybe_unused]] const uint32v16* __restrict__ src, uint64_t* __restrict__ timingResult, uint32_t waveSize, uint32_t numElements, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_SINGLE_BLOCK);

    const uint32_t numWaves = blockDim.x / waveSize;
    uint32_t waveId = threadIdx.x / waveSize;

    #ifdef __HIP_PLATFORM_AMD__
    // unlike inline asm, the builtin marks the result wave-uniform, so the addresses stay in SGPRs as s_load requires
    waveId = __builtin_amdgcn_readfirstlane(waveId);
    #endif

    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (uint32_t i = waveId; i < numElements; i += numWaves)
        {
            #ifdef __HIP_PLATFORM_AMD__
            amdScalarLoadBatch<LoadsPerBatch>(src + i);
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }
}


// The kernel instance for a number of loads per batch
static auto sL1ReadBandwidthKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:    return sL1ReadBandwidthKernel<1>;
        case 2:    return sL1ReadBandwidthKernel<2>;
        case 4:    return sL1ReadBandwidthKernel<4>;
        case 8:    return sL1ReadBandwidthKernel<8>;
        case 16:   return sL1ReadBandwidthKernel<16>;
        case 32:   return sL1ReadBandwidthKernel<32>;
        case 64:   return sL1ReadBandwidthKernel<64>;
        case 128:  return sL1ReadBandwidthKernel<128>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sL1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const uint32_t numElements = arraySizeBytes / sizeof(uint32v16);
    const auto kernel = sL1ReadBandwidthKernelFor(loadsPerBatch);

    uint32v16 *d_srcArr = util::allocateGPUMemory<uint32v16>((arraySizeBytes + BATCH_PADDING_BYTES) / sizeof(uint32v16));
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    double timeS = 0.0;

    // warm up; a block size that needs more registers than available does not launch
    kernel<<<1, numThreads>>>(d_srcArr, d_timingResult, util::getWarpSize(), numElements, WARMUP_REPS);
    const hipError_t launchError = hipGetLastError();

    if (launchError != hipErrorLaunchOutOfResources)
    {
        util::hipCheck(launchError);

        kernel<<<1, numThreads>>>(d_srcArr, d_timingResult, util::getWarpSize(), numElements, reps);
        util::hipCheck(hipGetLastError());
        util::hipCheck(hipStreamSynchronize(0));
        timeS = globalTimerSeconds(d_timingResult);
    }

    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) numElements * sizeof(uint32v16) * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}

namespace benchmark {
    namespace amd {
        CacheBandwidthResult measureScalarL1ReadBandwidthSweep(size_t arraySizeBytes) 
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

                    double bestBandwidth = 0.0;

                    for (uint32_t loadsPerBatch = 1; loadsPerBatch <= MAX_LOADS_PER_BATCH_SINGLE_BLOCK; loadsPerBatch *= 2)
                    {
                        auto [timeS, bandwidth] = sL1ReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps);

                        bestBandwidth = std::max(bestBandwidth, bandwidth);

                        if (bandwidth > result.measuredBandwidth)
                        {
                            result.measuredBandwidth = bandwidth;
                            result.time = timeS;
                            result.numThreads = numThreads;
                            result.numReps = reps;
                            result.loadsPerBatch = loadsPerBatch;
                        }
                    }

                    bandwidthResults.push_back(bestBandwidth);
                }

                result.bandwidthGridGiBs.push_back(bandwidthResults);
            }

            return result;
        }
    }
}
