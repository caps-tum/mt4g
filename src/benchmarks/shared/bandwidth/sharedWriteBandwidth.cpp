#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <numeric>

static constexpr auto MIN_REPS = 2;
static constexpr auto MAX_REPS = 262144; // 2 ^ 18

static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void sharedWriteBandwidthKernel(uint64_t* __restrict__ timings, uint32_t elementsPerThread, size_t reps) 
{
    const uint32_t tid = threadIdx.x;

    extern __shared__ uint32v4 memory[];

    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t sharedBaseAddr;
    asm volatile(
        "cvta.to.shared.u64 %0, %1;\n\t"
        : "=l"(sharedBaseAddr)
        : "l"(memory)
        : "memory"
    );
    #endif

    const uint32_t base = tid * elementsPerThread * 16u;

    uint32v4 dummy = {tid, tid + 1, tid + 2, tid + 3};

    uint64_t start, end;

    __syncthreads();

    if (tid == 0) 
    {
        #ifdef __HIP_PLATFORM_AMD__
        __asm__ volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_memtime %0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(start)
            :
            : "memory"
        );
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        __asm__ volatile(
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
        for (uint32_t i = 0; i < elementsPerThread; ++i)
        {
            const uint32_t offset = base + i * 16u;

            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "ds_write_b128 %0, %1\n\t"
                :
                : "v"(offset), "v"(dummy)
                : "memory"
            );
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            const uint64_t addr = sharedBaseAddr + static_cast<uint64_t>(offset);
            __asm__ volatile(
                "st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n\t"
                :
                : "l"(addr)
                , "r"(dummy.x)
                , "r"(dummy.y)
                , "r"(dummy.z)
                , "r"(dummy.w)
                : "memory"
            );
            #endif
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "s_waitcnt lgkmcnt(0)\n\t" 
        :
        :
        : "memory"
    );
    #endif
    
    __syncthreads();

    if (tid == 0) 
    {
        #ifdef __HIP_PLATFORM_AMD__
        __asm__ volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_memtime %0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(end)
            :
            : "memory"
        );
        *timings = end - start;
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        __asm__ volatile(
            "mov.u64 %0, %%clock64;\n\t"
            : "=l"(end)
            :
            : "memory"
        );
        *timings = end - start;
        #endif
    }
}

static std::tuple<uint64_t, double, double> sharedWriteBandwidthLauncher(uint32_t arraySizeBytes, uint32_t numThreads, size_t reps) 
{
    uint32_t numElements = arraySizeBytes / sizeof(uint32v4);
    uint32_t elementsPerThread = numElements / numThreads;

    uint64_t* d_timings = util::allocateGPUMemory<uint64_t>(1);

    // dynamic memory allocation
    sharedWriteBandwidthKernel<<<1, numThreads, arraySizeBytes>>>(d_timings, elementsPerThread, reps);

    std::vector<uint64_t> t = util::copyFromDevice<uint64_t>(d_timings, 1);

    uint64_t cycles = t[0];

    double gpuClockHz = util::getDeviceProperties().clockRate * 1000.0;
    double timeS = (double) cycles / gpuClockHz;
    double dataGiB = (double) arraySizeBytes  * reps / (1 * GiB);
 
    return {cycles, timeS, dataGiB / timeS};
}

namespace benchmark {
    double measureSharedWriteBandwidth(uint32_t arraySizeBytes)
    {
        std::vector<double> results(ROUNDS);

        uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;
        size_t maxReps = MAX_REPS;

        for (uint32_t i = 0; i < ROUNDS; ++i) 
        {
            results[i] = std::get<2>(sharedWriteBandwidthLauncher(arraySizeBytes, maxNumThreads, maxReps));
        }

        return util::average(results);
    }

    CacheBandwidthResult measureSharedWriteBandwidthSweep(uint32_t arraySizeBytes) 
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

                auto [cycles, timeS, bandwidth] = sharedWriteBandwidthLauncher(arraySizeBytes, numThreads, reps);
                
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
