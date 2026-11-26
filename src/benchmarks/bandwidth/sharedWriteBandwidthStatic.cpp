#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "benchmarks/bandwidth/bandwidthSweep.hpp"
#include "utils/util.hpp"

#include <vector>
#include <numeric>

static constexpr auto SIZE = 32 * KiB;
static constexpr auto NUM_ELEMENTS = SIZE / 16;
static constexpr auto WARMUP_REPS = 128;


// Same as the dynamic shared write kernel, but with a statically sized array
__global__ void sharedWriteBandwidthStaticKernel(uint64_t* __restrict__ timingResult, size_t reps)
{
    const uint32_t tid = threadIdx.x;
    const uint32v4 data = {tid, tid + 1, tid + 2, tid + 3};
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    __shared__ uint32v4 memory[NUM_ELEMENTS];

    #ifdef __HIP_PLATFORM_AMD__
    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)); // byte address in the LDS
    #endif

    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t base;
    asm volatile ("cvta.to.shared.u64 %0, %1;" : "=l"(base) : "l"(memory));
    #endif

    uint64_t start = 0;

    __syncthreads();
    if (tid == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (uint32_t i = tid; i < NUM_ELEMENTS; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile ("ds_write_b128 %0, %1\n\t" : : "v"(base + i * 16u), "v"(data) : "memory");
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile (
                "st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n\t"
                :
                : "l"(base + i * 16u), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w)
                : "memory"
            );
            #endif
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile ("s_waitcnt lgkmcnt(0)\n\t" : : : "memory"); // the compiler does not track the asm stores
    #endif

    __syncthreads();
    if (tid == 0) { *timingResult = __globaltimer() - start; }
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sharedWriteBandwidthStaticLauncher(uint32_t numThreads, size_t reps)
{
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    double timeS = 0.0;

    // warm up; a block size that needs more registers than available does not launch
    sharedWriteBandwidthStaticKernel<<<1, numThreads>>>(d_timingResult, WARMUP_REPS);
    const hipError_t launchError = hipGetLastError();

    if (launchError != hipErrorLaunchOutOfResources)
    {
        util::hipCheck(launchError);

        sharedWriteBandwidthStaticKernel<<<1, numThreads>>>(d_timingResult, reps);
        util::hipCheck(hipGetLastError());
        util::hipCheck(hipStreamSynchronize(0));
        timeS = globalTimerSeconds(d_timingResult);
    }

    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) SIZE * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}

namespace benchmark {
    CacheBandwidthResult measureSharedWriteBandwidthStaticSweep() 
    {
        uint32_t minNumThreads = util::getWarpSize();
        uint32_t maxNumThreads = util::getMaxThreadsPerBlock();
        size_t minReps = MIN_REPS;
        size_t maxReps = MAX_REPS;

        CacheBandwidthResult result(SIZE, 1);

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

                auto [timeS, bandwidth] = sharedWriteBandwidthStaticLauncher(numThreads, reps);
                
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
