#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>

static constexpr auto WARMUP_REPS = MAX_LOADS_PER_BATCH_SINGLE_BLOCK;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0, "loads per batch must divide the repetitions");


template <uint32_t LoadsPerBatch>
__global__ void sharedReadBandwidthKernel(uint32_t* __restrict__ dst, uint64_t* __restrict__ timingResult, uint32_t numElements, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_SINGLE_BLOCK);
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    extern __shared__ uint32v4 memory[];

    for (uint32_t i = threadIdx.x; i < numElements + SHARED_BATCH_PADDING_BYTES / sizeof(uint32v4); i += blockStride)
    {
        memory[i] = {0, 0, 0, 0};
    }

    #ifdef __HIP_PLATFORM_AMD__
    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)); // byte address in the LDS
    #endif

    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t base;
    asm volatile ("cvta.to.shared.u64 %0, %1;" : "=l"(base) : "l"(memory));
    #endif

    uint32_t dummy = 0;
    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (uint32_t i = threadIdx.x; i < numElements; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            amdSharedLoadBatch<LoadsPerBatch>(base + i * sizeof(uint32v4));
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            dummy ^= nvidiaSharedLoadBatch<LoadsPerBatch>(base + i * sizeof(uint32v4));
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


// The kernel instance for a number of loads per batch
static auto sharedReadBandwidthKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:    return sharedReadBandwidthKernel<1>;
        case 2:    return sharedReadBandwidthKernel<2>;
        case 4:    return sharedReadBandwidthKernel<4>;
        case 8:    return sharedReadBandwidthKernel<8>;
        case 16:   return sharedReadBandwidthKernel<16>;
        case 32:   return sharedReadBandwidthKernel<32>;
        case 64:   return sharedReadBandwidthKernel<64>;
        case 128:  return sharedReadBandwidthKernel<128>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sharedReadBandwidthLauncher(uint32_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const uint32_t numElements = arraySizeBytes / sizeof(uint32v4);
    const auto kernel = sharedReadBandwidthKernelFor(loadsPerBatch);

    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    double timeS = 0.0;

    // warm up; a block size that needs more registers than available does not launch
    kernel<<<1, numThreads, arraySizeBytes + SHARED_BATCH_PADDING_BYTES>>>(d_dstArr, d_timingResult, numElements, WARMUP_REPS);
    const hipError_t launchError = hipGetLastError();

    if (launchError != hipErrorLaunchOutOfResources)
    {
        util::hipCheck(launchError);

        kernel<<<1, numThreads, arraySizeBytes + SHARED_BATCH_PADDING_BYTES>>>(d_dstArr, d_timingResult, numElements, reps);
        util::hipCheck(hipGetLastError());
        util::hipCheck(hipStreamSynchronize(0));
        timeS = globalTimerSeconds(d_timingResult);
    }

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}

namespace benchmark 
{
    CacheBandwidthResult measureSharedReadBandwidthSweep(uint32_t arraySizeBytes) 
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
                    auto [timeS, bandwidth] = sharedReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps);

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
