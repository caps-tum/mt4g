#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>

static constexpr auto WARMUP_REPS = 8;


static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

static constexpr size_t MIN_EXPECTED_SIZE = 8192;// 8 * KiB, same assumption the Constant L1.5 Size benchmark makes
static constexpr auto MAX_ALLOWED_SIZE = MAX_ALLOWED_INDEX * sizeof(uint32_t);// 63 KiB of the 64 KiB constant array

// Minimum stride that places each load on a different constant cache line.
// Current NVIDIA constant cache lines are 64B; larger strides show no further change.
static constexpr size_t MIN_LINE_SKIP_STRIDE = 64;

// Constant L1.5 bandwidth benchmark for a single SM. Uses ld.const on the
// **constant** array, with each thread striding by one fetch granularity so
// every load targets a fresh cache line. This avoids the L1/L1.5 mixture caused
// by contiguous loads. The access pattern matches the Constant L1.5 Size and
// Latency benchmarks; per-access eviction is avoided because it would serialize
// the loop and measure eviction cost rather than streaming bandwidth.
__global__ void constantL15ReadBandwidthKernel(uint32_t* __restrict__ dst, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps, size_t strideElements)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t* base = arr16384AscStride0 + tid * elementsPerThread * strideElements;

    uint32_t dummy = 0;

    // Warm up the constant caches
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
                : "l"(base + i * strideElements)
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
                : "l"(base + i * strideElements)
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


static std::tuple<uint64_t, double, double> constantL15ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, size_t strideBytes)
{
    size_t strideElements = strideBytes / sizeof(uint32_t);
    size_t totalElements = arraySizeBytes / sizeof(uint32_t);
    // One load per stride, so the working set spans elementsPerThread * stride
    // elements per thread while only elementsPerThread of them are read.
    size_t elementsPerThread = totalElements / numThreads / strideElements;

    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    // Run the kernel
    constantL15ReadBandwidthKernel<<<1, numThreads>>>(d_dstArr, d_timingResult, elementsPerThread, reps, strideElements);

    // Get the timings from the device
    std::vector<uint64_t> timingResult = util::copyFromDevice<uint64_t>(d_timingResult, 1);

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    // Constant working sets are small, so integer division can leave bytes untouched.
    // Count only the bytes actually read, not the full cache lines fetched by the stride.
    double gpuClockHz = util::getDeviceProperties().clockRate * 1000;
    double dataGiB = (double) (elementsPerThread * numThreads * sizeof(uint32_t)) * reps / (1 * GiB);
    double timeS = (double) timingResult[0] / gpuClockHz;

    // return (cycles, time in seconds, measured bandwidth)
    return {timingResult[0], timeS, dataGiB / timeS};
}


// Constant memory is limited to 64 KiB, so cap the requested size.
// Keep the working set above MIN_EXPECTED_SIZE to avoid fitting in constant L1.
static size_t capConstantArraySize(size_t arraySizeBytes)
{
    if (arraySizeBytes > MAX_ALLOWED_SIZE) {
        std::cerr << "WARNING: Constant L1.5 Read Bandwidth requested " << arraySizeBytes
                  << " Bytes of constant data, capping to " << MAX_ALLOWED_SIZE << " Bytes" << std::endl;
        arraySizeBytes = MAX_ALLOWED_SIZE;
    }
    if (arraySizeBytes < MIN_EXPECTED_SIZE) {
        std::cerr << "WARNING: Constant L1.5 Read Bandwidth working set of " << arraySizeBytes
                  << " Bytes may still fit into the constant L1, results may reflect the L1 instead" << std::endl;
    }

    return arraySizeBytes;
}

// Use at least one full constant cache line per stride to avoid L1 reuse and isolate L1.5 traffic.
static size_t capStride(size_t strideBytes)
{
    strideBytes = std::max(strideBytes, MIN_LINE_SKIP_STRIDE);

    return strideBytes - (strideBytes % sizeof(uint32_t));
}

// Bound threads by the number of distinct cache lines available.
// Each thread gets totalElements / strideElements lines.
static uint32_t capNumThreads(size_t arraySizeBytes, size_t strideBytes)
{
    size_t lines = (arraySizeBytes / sizeof(uint32_t)) / (strideBytes / sizeof(uint32_t));
    uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;

    return static_cast<uint32_t>(std::min(static_cast<size_t>(maxNumThreads), lines));
}


namespace benchmark
{
    namespace nvidia
    {
        double measureConstantL15ReadBandwidth(size_t arraySizeBytes, size_t constantFetchGranularityBytes)
        {
            arraySizeBytes = capConstantArraySize(arraySizeBytes);
            size_t strideBytes = capStride(constantFetchGranularityBytes);

            uint32_t maxNumThreads = capNumThreads(arraySizeBytes, strideBytes);
            size_t maxReps = MAX_REPS;

            if (maxNumThreads == 0) {
                std::cerr << "WARNING: Constant L1.5 Read Bandwidth working set too small to benchmark, skipping" << std::endl;
                return 0.0;
            }

            std::vector<double> results(ROUNDS);

            for (uint32_t i = 0; i < ROUNDS; ++i)
            {
                results[i] = std::get<2>(constantL15ReadBandwidthLauncher(arraySizeBytes, maxNumThreads, maxReps, strideBytes));
            }

            return util::average(results);
        }

        CacheBandwidthResult measureConstantL15ReadBandwidthSweep(size_t arraySizeBytes, size_t constantFetchGranularityBytes)
        {
            arraySizeBytes = capConstantArraySize(arraySizeBytes);
            size_t strideBytes = capStride(constantFetchGranularityBytes);

            uint32_t minNumThreads = util::getDeviceProperties().warpSize;
            uint32_t maxNumThreads = capNumThreads(arraySizeBytes, strideBytes);
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
                std::cerr << "WARNING: Constant L1.5 Read Bandwidth working set too small for a full warp, skipping" << std::endl;
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

                    auto [cycles, timeS, bandwidth] = constantL15ReadBandwidthLauncher(arraySizeBytes, numThreads, reps, strideBytes);

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
