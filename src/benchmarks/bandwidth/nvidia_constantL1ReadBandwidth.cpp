#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>

static constexpr auto WARMUP_REPS = 8;


static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

static constexpr auto MAX_ALLOWED_SIZE = MAX_ALLOWED_INDEX * sizeof(uint32_t);// 63 KiB of the 64 KiB constant array

// Constant L1 bandwidth benchmark for a single SM. Uses ld.const on the shared
// constant array; the caller provides a working set that fits in constant L1.
// Scalar loads are used because the uint32_t array has no guaranteed 16-byte alignment.
__global__ void constantL1ReadBandwidthKernel(uint32_t* __restrict__ dst, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t* base = arr16384AscStride0 + tid * elementsPerThread;

    uint32_t dummy = 0;

    // Warm up the constant cache
    for (size_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            uint32_t loaded = 0;

            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile (
                "{\n\t"
                ".reg .u64 cp;\n\t"
                "cvta.to.const.u64 cp, %1;\n\t"
                "ld.const.u32 %0, [cp];\n\t"
                "}"
                : "=r"(loaded)
                : "l"(base + i)
                : "memory"
            );
            #endif

            dummy ^= loaded;
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
            uint32_t loaded = 0;

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile (
                "{\n\t"
                ".reg .u64 cp;\n\t"
                "cvta.to.const.u64 cp, %1;\n\t"
                "ld.const.u32 %0, [cp];\n\t"
                "}"
                : "=r"(loaded)
                : "l"(base + i)
                : "memory"
            );
            #endif

            dummy ^= loaded;
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


static std::tuple<uint64_t, double, double> constantL1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps)
{
    size_t totalElements = arraySizeBytes / sizeof(uint32_t);
    size_t elementsPerThread = totalElements / numThreads;

    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    // Run the kernel
    constantL1ReadBandwidthKernel<<<1, numThreads>>>(d_dstArr, d_timingResult, elementsPerThread, reps);

    // Get the timings from the device
    std::vector<uint64_t> timingResult = util::copyFromDevice<uint64_t>(d_timingResult, 1);

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    // Constant working sets are small, so integer division can leave bytes untouched.
    // Use the bytes actually read rather than the requested array size.
    double gpuClockHz = util::getDeviceProperties().clockRate * 1000;
    double dataGiB = (double) (elementsPerThread * numThreads * sizeof(uint32_t)) * reps / (1 * GiB);
    double timeS = (double) timingResult[0] / gpuClockHz;

    // return (cycles, time in seconds, measured bandwidth)
    return {timingResult[0], timeS, dataGiB / timeS};
}


// Constant memory is limited to 64 KiB and L1 is only a few KiB.
// Cap size and threads so every thread reads at least one element.
static size_t capConstantArraySize(size_t arraySizeBytes, const char* benchmarkName)
{
    if (arraySizeBytes > MAX_ALLOWED_SIZE) {
        std::cerr << "WARNING: " << benchmarkName << " requested " << arraySizeBytes
                  << " Bytes of constant data, capping to " << MAX_ALLOWED_SIZE << " Bytes" << std::endl;
        arraySizeBytes = MAX_ALLOWED_SIZE;
    }

    return arraySizeBytes;
}

static uint32_t capNumThreads(size_t arraySizeBytes)
{
    size_t totalElements = arraySizeBytes / sizeof(uint32_t);
    uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;

    return static_cast<uint32_t>(std::min(static_cast<size_t>(maxNumThreads), totalElements));
}


namespace benchmark
{
    namespace nvidia
    {
        double measureConstantL1ReadBandwidth(size_t arraySizeBytes)
        {
            arraySizeBytes = capConstantArraySize(arraySizeBytes, "Constant L1 Read Bandwidth");

            uint32_t maxNumThreads = capNumThreads(arraySizeBytes);
            size_t maxReps = MAX_REPS;

            if (maxNumThreads == 0) {
                std::cerr << "WARNING: Constant L1 Read Bandwidth working set too small to benchmark, skipping" << std::endl;
                return 0.0;
            }

            std::vector<double> results(ROUNDS);

            for (uint32_t i = 0; i < ROUNDS; ++i)
            {
                results[i] = std::get<2>(constantL1ReadBandwidthLauncher(arraySizeBytes, maxNumThreads, maxReps));
            }

            return util::average(results);
        }

        CacheBandwidthResult measureConstantL1ReadBandwidthSweep(size_t arraySizeBytes)
        {
            arraySizeBytes = capConstantArraySize(arraySizeBytes, "Constant L1 Read Bandwidth");

            uint32_t minNumThreads = util::getDeviceProperties().warpSize;
            uint32_t maxNumThreads = capNumThreads(arraySizeBytes);
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

            if (maxNumThreads < minNumThreads) {
                std::cerr << "WARNING: Constant L1 Read Bandwidth working set too small for a full warp, skipping" << std::endl;
                return result;
            }

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

                    auto [cycles, timeS, bandwidth] = constantL1ReadBandwidthLauncher(arraySizeBytes, numThreads, reps);

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
