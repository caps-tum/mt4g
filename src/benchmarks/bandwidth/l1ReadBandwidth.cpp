#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>

static constexpr auto WARMUP_REPS = 128;

#ifdef __HIP_PLATFORM_AMD__

static constexpr auto MS_PER_SECOND = 1000.0; // ms
static constexpr uint32_t NUM_BLOCKS = 1;

static constexpr size_t LOADS_PER_GROUP = 8;
static_assert(WARMUP_REPS % LOADS_PER_GROUP == 0 && MIN_REPS % LOADS_PER_GROUP == 0,
              "rep counts must be multiples of LOADS_PER_GROUP");

static constexpr size_t GROUP_LOAD_STRIDE = 0;

__global__ void l1ReadBandwidthKernel(uint32v4* __restrict__ dst, const uint32v4* __restrict__ src, size_t totalElements, size_t reps, size_t groupLoadStride)
{
    const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint32v4 dummy {0, 0, 0, 0};

    for (size_t rep = 0; rep < reps; rep += LOADS_PER_GROUP)
    {
        for (size_t i = gtid; i < totalElements; i += stride)
        {
            #pragma unroll
            for (size_t k = 0; k < LOADS_PER_GROUP; ++k)
            {
                const uint32v4 loaded = src[i + k * groupLoadStride];

                dummy.x ^= loaded.x;
                dummy.y ^= loaded.y;
                dummy.z ^= loaded.z;
                dummy.w ^= loaded.w;
            }
        }
    }

    dst[gtid] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> l1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, hipStream_t stream) 
{
    const size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    const size_t totalThreads = static_cast<size_t>(NUM_BLOCKS) * numThreads;

    // Allocate device arrays
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(totalElements);
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(totalThreads);

    // Warm up
    l1ReadBandwidthKernel<<<NUM_BLOCKS, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, WARMUP_REPS, GROUP_LOAD_STRIDE);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    l1ReadBandwidthKernel<<<NUM_BLOCKS, numThreads, 0, stream>>>(d_dstArr, d_srcArr, totalElements, reps, GROUP_LOAD_STRIDE);
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
        // pin the entire sweep to a single CU.
        auto stream = util::createStreamForCU(0);

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
        result.numBlocks = NUM_BLOCKS;
        result.numReps = 0;
        result.blocksTested.push_back(NUM_BLOCKS);

        // Precompute full thread/rep axes for CSV alignment.
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
        result.bandwidth3D.assign(1, std::vector<std::vector<double>>(
            numThreadSteps, std::vector<double>(numRepSteps, 0.0)));

        // Measure every thread count and repetition count.
        for (size_t ti = 0; ti < numThreadSteps; ++ti)
        {
            const uint32_t numThreads = result.threadsTested[ti];

            for (size_t ri = 0; ri < numRepSteps; ++ri)
            {
                const size_t reps = result.repsTested[ri];

                auto [timeS, bandwidth] = l1ReadBandwidthLauncher(arraySizeBytes, numThreads, reps, stream);

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
#endif // __HIP_PLATFORM_AMD__

#ifdef __HIP_PLATFORM_NVIDIA__

__global__ void l1ReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps) 
{
    const uint32_t tid = threadIdx.x;
    const uint32v4* base = src + tid * elementsPerThread;

    uint32v4 dummy {0, 0, 0, 0};

    // Warm up L1
    for (size_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            uint32v4 loaded;

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
            uint32v4 loaded;

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
}
#endif // __HIP_PLATFORM_NVIDIA__
