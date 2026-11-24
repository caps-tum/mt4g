#pragma once

#include "benchmarks/base.hpp"
#include "config.hpp"
#include "typedef/cacheBandwidthResult.hpp"
#include "typedef/vectorTypes.hpp"
#include "utils/util.hpp"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

// Single-block read bandwidth benchmarks issue a batch of N independent loads to a small array before waiting for them.
// N is swept as powers of two up to this, beyond what the hardware keeps in flight, to see the plateau.
static constexpr uint32_t MAX_LOADS_PER_BATCH = 128;

// Load k of a batch reads k * 128 bytes further, so the loads of a batch hit different cache lines
// (AMD vector memory uses k % 32, as the offset field of the instruction has 12 bits). The array needs this much padding.
static constexpr size_t BATCH_STRIDE_BYTES = 128;
static constexpr size_t BATCH_PADDING_BYTES = MAX_LOADS_PER_BATCH * BATCH_STRIDE_BYTES;

// Calls f(std::integral_constant<uint32_t, N>) for the N known only at run time, e.g. to pick a kernel template instance:
// withLoadsPerBatch(n, [](auto n) { return kernel<decltype(n)::value>; }). Only N up to MaxLoadsPerBatch are instantiated.
template <uint32_t MaxLoadsPerBatch = MAX_LOADS_PER_BATCH, typename F>
auto withLoadsPerBatch(uint32_t loadsPerBatch, F f)
{
    switch (loadsPerBatch)
    {
        case 1:   return f(std::integral_constant<uint32_t, 1>{});
        case 2:   return f(std::integral_constant<uint32_t, 2>{});
        case 4:   return f(std::integral_constant<uint32_t, 4>{});
        case 8:   return f(std::integral_constant<uint32_t, 8>{});
        case 16:  return f(std::integral_constant<uint32_t, 16>{});
        case 32:  if constexpr (MaxLoadsPerBatch >= 32)  { return f(std::integral_constant<uint32_t, 32>{}); } break;
        case 64:  if constexpr (MaxLoadsPerBatch >= 64)  { return f(std::integral_constant<uint32_t, 64>{}); } break;
        case 128: if constexpr (MaxLoadsPerBatch >= 128) { return f(std::integral_constant<uint32_t, 128>{}); } break;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}

#define LOADS_PER_BATCH_CHECK(N, MAX) \
    static_assert((N) <= (MAX) && ((N) & ((N) - 1)) == 0, "loads per batch must be a power of two up to " #MAX)


// Runs launch(reps) for the warm-up and the measurement of a single-block kernel that writes its own duration
// (__globaltimer() ticks) to d_ticks. Returns the time in seconds, or 0 if the configuration does not launch
// because it needs more resources (registers) than this block size allows.
template <typename Launch>
double timeSingleBlockKernel(uint64_t *d_ticks, size_t warmupReps, size_t reps, hipStream_t stream, Launch launch)
{
    launch(warmupReps);
    const hipError_t launchError = hipGetLastError();
    if (launchError == hipErrorLaunchOutOfResources) { return 0.0; }
    util::hipCheck(launchError);

    launch(reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipStreamSynchronize(stream));

    const uint64_t ticks = util::copyFromDevice(d_ticks, 1)[0];
    return (double) ticks / (util::getWallClockRateKHz() * 1000.0);
}


// Sweeps threads x reps x loads per batch with a single block. measure(numThreads, loadsPerBatch, reps) returns
// {time in seconds, bandwidth in GiB/s}; each grid cell keeps the best number of loads per batch.
// Write benchmarks pass maxLoadsPerBatch = 1, as stores do not wait for each other.
template <typename Measure>
CacheBandwidthResult sweepSingleBlockBandwidth(size_t dataBytes, Measure measure, uint32_t maxLoadsPerBatch = MAX_LOADS_PER_BATCH)
{
    uint32_t minNumThreads = util::getWarpSize();
    uint32_t maxNumThreads = util::getMaxThreadsPerBlock();

    CacheBandwidthResult result{};
    result.measuredBandwidth = 0.0;
    result.dataBytes = dataBytes;
    result.cycles = 0;
    result.time = 0.0;
    result.numThreads = 0;
    result.numBlocks = 1;
    result.numReps = 0;

    for (uint32_t numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads *= 2)
    {
        std::vector<double> bandwidthResults;

        result.threadsTested.push_back(numThreads);

        for (size_t reps = MIN_REPS; reps <= MAX_REPS; reps *= 2)
        {
            if (numThreads == minNumThreads)
            {
                result.repsTested.push_back(reps);
            }

            double bestBandwidth = 0.0;

            for (uint32_t loadsPerBatch = 1; loadsPerBatch <= maxLoadsPerBatch; loadsPerBatch *= 2)
            {
                auto [timeS, bandwidth] = measure(numThreads, loadsPerBatch, reps);

                bestBandwidth = std::max(bestBandwidth, bandwidth);

                if (bandwidth > result.measuredBandwidth)
                {
                    result.measuredBandwidth = bandwidth;
                    result.time = timeS;
                    result.numThreads = numThreads;
                    result.numReps = reps;
                    result.loadsPerBatch = maxLoadsPerBatch > 1 ? loadsPerBatch : 0;
                }
            }

            bandwidthResults.push_back(bestBandwidth);
        }

        result.bandwidthGridGiBs.push_back(bandwidthResults);
    }

    return result;
}

#ifdef __HIP_PLATFORM_AMD__
// AMD_BATCH_<N>(LOAD) is the asm text of N loads, LOAD(offset) gives one load at a byte offset
#define AMD_BATCH_1(LOAD)   LOAD(0)
#define AMD_BATCH_2(LOAD)   AMD_BATCH_1(LOAD) LOAD(128)
#define AMD_BATCH_4(LOAD)   AMD_BATCH_2(LOAD) LOAD(256) LOAD(384)
#define AMD_BATCH_8(LOAD)   AMD_BATCH_4(LOAD) LOAD(512) LOAD(640) LOAD(768) LOAD(896)
#define AMD_BATCH_16(LOAD)  AMD_BATCH_8(LOAD) LOAD(1024) LOAD(1152) LOAD(1280) LOAD(1408) \
                            LOAD(1536) LOAD(1664) LOAD(1792) LOAD(1920)
#define AMD_BATCH_32(LOAD)  AMD_BATCH_16(LOAD) LOAD(2048) LOAD(2176) LOAD(2304) LOAD(2432) \
                            LOAD(2560) LOAD(2688) LOAD(2816) LOAD(2944) \
                            LOAD(3072) LOAD(3200) LOAD(3328) LOAD(3456) \
                            LOAD(3584) LOAD(3712) LOAD(3840) LOAD(3968)
#define AMD_BATCH_64(LOAD)  AMD_BATCH_32(LOAD) AMD_BATCH_32(LOAD)
#define AMD_BATCH_128(LOAD) AMD_BATCH_64(LOAD) AMD_BATCH_64(LOAD)

// One asm with the N loads of a batch and the wait for them (N a template parameter). The loads must write registers
// listed in the clobbers, so the compiler cannot reuse them while the loads are in flight.
#define AMD_LOAD_BATCH(N, LOAD, WAIT, ...) \
    if constexpr ((N) == 1)   { __asm__ volatile (AMD_BATCH_1(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 2)   { __asm__ volatile (AMD_BATCH_2(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 4)   { __asm__ volatile (AMD_BATCH_4(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 8)   { __asm__ volatile (AMD_BATCH_8(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 16)  { __asm__ volatile (AMD_BATCH_16(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 32)  { __asm__ volatile (AMD_BATCH_32(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 64)  { __asm__ volatile (AMD_BATCH_64(LOAD) WAIT __VA_ARGS__); } \
    if constexpr ((N) == 128) { __asm__ volatile (AMD_BATCH_128(LOAD) WAIT __VA_ARGS__); }
#endif


// Shared memory (LDS) batches. Load k reads k * 128 bytes further on AMD (k % 32) and k * 16 bytes further on NVIDIA,
// both without bank conflicts; the shared array needs this much padding.
static constexpr uint32_t SHARED_BATCH_PADDING_BYTES = 4096;

#ifdef __HIP_PLATFORM_AMD__
#define AMD_SHARED_LOAD(offset) "ds_read_b128 v[20:23], %0 offset:" #offset "\n\t"

// All loads write v20-v23 (the data is not used); address is the byte address in the LDS
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ void amdSharedLoadBatch(uint32_t address)
{
    AMD_LOAD_BATCH(LoadsPerBatch, AMD_SHARED_LOAD, "s_waitcnt lgkmcnt(0)\n\t", : : "v"(address) : "memory", "v20", "v21", "v22", "v23")
}
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
// All four words are used and all addresses differ, otherwise the compiler narrows or merges the loads;
// address is the address in the shared state space (cvta.to.shared)
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ uint32_t nvidiaSharedLoadBatch(uint64_t address)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        uint32_t x, y, z, w;
        asm volatile (
            "ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
            : "l"(address + k * 16)
            : "memory"
        );
        dummy ^= x ^ y ^ z ^ w;
    }

    return dummy;
}
#endif
