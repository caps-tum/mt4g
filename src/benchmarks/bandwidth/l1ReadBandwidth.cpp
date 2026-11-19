#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>
#include <limits>

static constexpr auto WARMUP_REPS = 128;
static constexpr auto MS_PER_SECOND = 1000.0; // ms


__global__ void l1ReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps) 
{
    const uint32_t tid = threadIdx.x;
    const uint32v4* base = src + tid * elementsPerThread;

    #ifdef __HIP_PLATFORM_AMD__
    const uint64_t addr0 = reinterpret_cast<uint64_t>(base);
    #endif

    uint32v4 dummy {0, 0, 0, 0};

    // Warm up L1
    for (size_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            uint32v4 loaded;
            #ifdef __HIP_PLATFORM_AMD__
            asm volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr0 + i * sizeof(uint32v4))
                : "memory"
            );
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile (
                "ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(loaded.x)
                , "=r"(loaded.y)
                , "=r"(loaded.z)
                , "=r"(loaded.w)
                : "l"(base + i)
                : "memory"
            );
            #endif

            dummy.x ^= loaded.x;
        }
    }

    uint64_t start, end;

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "s_waitcnt vmcnt(0)\n\t"
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
            : "=s"(start)
            :
            : "memory"
        );
        #endif

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
            uint32v4 loaded;

            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr0 + i * sizeof(uint32v4))
                : "memory"
            );
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile (
                "ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(loaded.x)
                , "=r"(loaded.y)
                , "=r"(loaded.z)
                , "=r"(loaded.w)
                : "l"(base + i)
                : "memory"
            );
            #endif

            dummy.x ^= loaded.x;
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "s_waitcnt vmcnt(0)\n\t"
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

        *timing_result = end - start;
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        __asm__ volatile (
            "mov.u64 %0, %%clock64;\n\t"
            : "=l"(end)
            :
            : "memory"
        );

        *timing_result = end - start;
        #endif
    }

    dst[tid] = dummy; // prevent dead code elimination
}


static std::tuple<uint64_t, double, double> l1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps) 
{
    size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    size_t elementsPerThread = totalElements / numThreads;
           
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(totalElements);
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    // Run the kernel
    l1ReadBandwidthKernel<<<1, numThreads>>>(d_dstArr, d_srcArr, d_timingResult, elementsPerThread, reps);

    // Get the timings from the device
    std::vector<uint64_t> timingResult = util::copyFromDevice<uint64_t>(d_timingResult, 1);

    // calculate the bandwidth
    double gpuClockHz = util::getClockRateKHz() * 1000.0;
    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    double timeS = (double) timingResult[0] / gpuClockHz;
    
    // return (cycles, time in seconds, measured bandwidth)
    return {timingResult[0], timeS, dataGiB / timeS};
}


__global__ void amdL1ReadBandwidthKernel(uint32v4* __restrict__ dst, const uint32v4* __restrict__ src, size_t totalElements, size_t reps) 
{
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint32v4 dummy {0, 0, 0, 0};

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            uint32v4 loaded;

            #ifdef __HIP_PLATFORM_AMD__
            const uint64_t addr = reinterpret_cast<uint64_t>(src + i);

            __asm__ volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr)
                : "memory"
            );
            #endif

            dummy.x ^= loaded.x;
        }
    }

    dst[gtid] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> amdL1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, hipStream_t stream) 
{
    const size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    const size_t totalThreads = static_cast<size_t>(numBlocks) * numThreads;

    // Allocate device arrays
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(totalElements);
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(totalThreads);

    // Warm up
    amdL1ReadBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, WARMUP_REPS);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    amdL1ReadBandwidthKernel<<<numBlocks, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, reps);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark 
{
    CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes) 
    {
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

                auto [cycles, timeS, bandwidth] = l1ReadBandwidthLauncher(arraySizeBytes, numThreads, reps);
                
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

    namespace amd
    {
        CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes)
        {
            auto stream = util::createStreamForCU(0);

            const uint32_t numBlocks = 1;
            uint32_t minThreads = util::getWarpSize();
            uint32_t maxThreads = util::getMaxThreadsPerBlock();

            size_t minReps = MIN_REPS;
            size_t maxReps = MAX_REPS;

            CacheBandwidthResult result{};
            result.measuredBandwidth = 0.0;
            result.dataBytes = arraySizeBytes;
            result.cycles = 0;
            result.time = 0.0;
            result.numThreads = 0;
            result.numBlocks = numBlocks;
            result.numReps = 0;

            // Precompute full thread/rep axes for CSV alignment.
            result.blocksTested.push_back(numBlocks);
            for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
            {
                result.threadsTested.push_back(numThreads);
            }
            for (size_t reps = minReps; reps <= maxReps; reps *= 2)
            {
                result.repsTested.push_back(reps);
            }

            const size_t numThreadSteps = result.threadsTested.size();
            const size_t numRepSteps = result.repsTested.size();
            const double UNMEASURED = std::numeric_limits<double>::quiet_NaN();

            result.bandwidth3D.assign(1, std::vector<std::vector<double>>(
                numThreadSteps, std::vector<double>(numRepSteps, UNMEASURED)));

            // Search thread counts from highest to lowest.
            for (size_t ti = numThreadSteps; ti-- > 0; )
            {
                const uint32_t numThreads = result.threadsTested[ti];

                for (size_t ri = 0; ri < numRepSteps; ++ri)
                {
                    const size_t reps = result.repsTested[ri];

                    auto [timeS, bandwidth] = amdL1ReadBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, stream);

                    result.bandwidth3D[0][ti][ri] = bandwidth;

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numReps = reps;
                    }
                }
            }

            util::hipCheck(hipStreamDestroy(stream));

            return result;
        }
    }
}
