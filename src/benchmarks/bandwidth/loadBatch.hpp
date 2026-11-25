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
#include <limits>

// Single-block read bandwidth benchmarks issue a batch of N independent loads to a small array before waiting for them.
// N is swept as powers of two up to this, beyond what the hardware keeps in flight, to see the plateau.
static constexpr uint32_t MAX_LOADS_PER_BATCH = 128;

// Load k of a batch reads k * 128 bytes further, so the loads of a batch hit different cache lines
// (AMD vector memory uses k % 32, as the offset field of the instruction has 12 bits). The array needs this much padding.
static constexpr size_t BATCH_STRIDE_BYTES = 128;
static constexpr size_t BATCH_PADDING_BYTES = MAX_LOADS_PER_BATCH * BATCH_STRIDE_BYTES;

// Multi-block read bandwidth benchmarks sweep fewer loads per batch (2-4 suffice today, the rest is headroom)
static constexpr uint32_t MAX_MULTI_BLOCK_LOADS_PER_BATCH = 16;

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


// Multi-block batches: load k reads element i + k * gridStride, so every element is still read once per rep.
// (The single-block layout would let neighbouring threads load the same lines within a batch, served by a closer cache.)
#ifdef __HIP_PLATFORM_AMD__
// AMD_INDEXED_<N>(LOAD) is the asm text of N loads, LOAD(k) gives the load of address operand k
#define AMD_INDEXED_1(LOAD)  LOAD(0)
#define AMD_INDEXED_2(LOAD)  AMD_INDEXED_1(LOAD) LOAD(1)
#define AMD_INDEXED_4(LOAD)  AMD_INDEXED_2(LOAD) LOAD(2) LOAD(3)
#define AMD_INDEXED_8(LOAD)  AMD_INDEXED_4(LOAD) LOAD(4) LOAD(5) LOAD(6) LOAD(7)
#define AMD_INDEXED_16(LOAD) AMD_INDEXED_8(LOAD) LOAD(8) LOAD(9) LOAD(10) LOAD(11) LOAD(12) LOAD(13) LOAD(14) LOAD(15)

#define AMD_ADDRESSES_1(a)  "v"(a[0])
#define AMD_ADDRESSES_2(a)  AMD_ADDRESSES_1(a), "v"(a[1])
#define AMD_ADDRESSES_4(a)  AMD_ADDRESSES_2(a), "v"(a[2]), "v"(a[3])
#define AMD_ADDRESSES_8(a)  AMD_ADDRESSES_4(a), "v"(a[4]), "v"(a[5]), "v"(a[6]), "v"(a[7])
#define AMD_ADDRESSES_16(a) AMD_ADDRESSES_8(a), "v"(a[8]), "v"(a[9]), "v"(a[10]), "v"(a[11]), "v"(a[12]), "v"(a[13]), "v"(a[14]), "v"(a[15])

#define AMD_GRID_LOAD(k)     "flat_load_dwordx4 v[20:23], %" #k "\n\t"
#define AMD_GRID_LOAD_GLC(k) "flat_load_dwordx4 v[20:23], %" #k " " GLC "\n\t"

#define AMD_GRID_BATCH(N, LOAD, a) \
    if constexpr ((N) == 1)  { __asm__ volatile (AMD_INDEXED_1(LOAD) "s_waitcnt vmcnt(0)\n\t" : : AMD_ADDRESSES_1(a) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 2)  { __asm__ volatile (AMD_INDEXED_2(LOAD) "s_waitcnt vmcnt(0)\n\t" : : AMD_ADDRESSES_2(a) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 4)  { __asm__ volatile (AMD_INDEXED_4(LOAD) "s_waitcnt vmcnt(0)\n\t" : : AMD_ADDRESSES_4(a) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 8)  { __asm__ volatile (AMD_INDEXED_8(LOAD) "s_waitcnt vmcnt(0)\n\t" : : AMD_ADDRESSES_8(a) : "memory", "v20", "v21", "v22", "v23"); } \
    if constexpr ((N) == 16) { __asm__ volatile (AMD_INDEXED_16(LOAD) "s_waitcnt vmcnt(0)\n\t" : : AMD_ADDRESSES_16(a) : "memory", "v20", "v21", "v22", "v23"); }

// All loads write v20-v23 (the data is not used); BypassL1 adds the GLC bit (L2 benchmark)
template <uint32_t LoadsPerBatch, bool BypassL1>
__device__ __forceinline__ void amdGridLoadBatch(const uint32v4 *src, size_t gridStride)
{
    uint64_t addresses[LoadsPerBatch];

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k) { addresses[k] = reinterpret_cast<uint64_t>(src + k * gridStride); }

    if constexpr (BypassL1) { AMD_GRID_BATCH(LoadsPerBatch, AMD_GRID_LOAD_GLC, addresses) }
    else                    { AMD_GRID_BATCH(LoadsPerBatch, AMD_GRID_LOAD, addresses) }
}
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
// All four words are used, otherwise the compiler narrows the loads; BypassL1 uses ld.global.cg (L2 benchmark)
template <uint32_t LoadsPerBatch, bool BypassL1>
__device__ __forceinline__ uint32_t nvidiaGridLoadBatch(const uint32v4 *src, size_t gridStride)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        uint32_t x, y, z, w;
        if constexpr (BypassL1)
        {
            asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(src + k * gridStride) : "memory");
        }
        else
        {
            asm volatile ("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(src + k * gridStride) : "memory");
        }
        dummy ^= x ^ y ^ z ^ w;
    }

    return dummy;
}
#endif

// Reads every element of src[0..n) once per rep with batches of N loads along the grid-stride loop, single loads for the tail
template <uint32_t LoadsPerBatch, bool BypassL1>
__device__ __forceinline__ uint32_t gridReadPass(const uint32v4 *src, size_t first, size_t n, size_t gridStride)
{
    uint32_t dummy = 0;
    size_t i = first;

    for (; i + (LoadsPerBatch - 1) * gridStride < n; i += LoadsPerBatch * gridStride)
    {
        #ifdef __HIP_PLATFORM_AMD__
        amdGridLoadBatch<LoadsPerBatch, BypassL1>(src + i, gridStride);
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        dummy ^= nvidiaGridLoadBatch<LoadsPerBatch, BypassL1>(src + i, gridStride);
        #endif
    }

    for (; i < n; i += gridStride)
    {
        #ifdef __HIP_PLATFORM_AMD__
        amdGridLoadBatch<1, BypassL1>(src + i, gridStride);
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        dummy ^= nvidiaGridLoadBatch<1, BypassL1>(src + i, gridStride);
        #endif
    }

    return dummy;
}


// Sweeps blocks x threads x reps x loads per batch. measure(numBlocks, numThreads, loadsPerBatch, reps) returns
// {time in seconds, bandwidth in GiB/s}; each grid cell keeps the best number of loads per batch.
// Block and thread counts are searched from highest to lowest: a >= 25 % drop from the best bandwidth of a block count
// ends its thread sweep and skips this and lower thread counts for the remaining block counts.
// Write benchmarks pass maxLoadsPerBatch = 1, as stores do not wait for each other.
template <typename Measure>
CacheBandwidthResult sweepMultiBlockBandwidth(size_t dataBytes, const std::vector<size_t> &repsTested, Measure measure, uint32_t maxLoadsPerBatch = MAX_MULTI_BLOCK_LOADS_PER_BATCH)
{
    uint32_t minThreads = util::getWarpSize();
    uint32_t maxThreads = util::getMaxThreadsPerBlock();

    uint32_t minBlocks = util::getNumberOfComputeUnits();
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getMaxBlocksPerMultiProcessor();

    CacheBandwidthResult result{};
    result.measuredBandwidth = 0.0;
    result.dataBytes = dataBytes;
    result.cycles = 0;
    result.time = 0.0;
    result.numThreads = 0;
    result.numBlocks = 0;
    result.numReps = 0;

    // Full block/thread/rep axes for CSV alignment; the sweep runs descending, the axes stay ascending
    for (uint32_t numBlocks = minBlocks; numBlocks <= maxBlocks; numBlocks *= 2)
    {
        result.blocksTested.push_back(numBlocks);
    }
    for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
    {
        result.threadsTested.push_back(numThreads);
    }
    result.repsTested = repsTested;

    const size_t numBlockSteps = result.blocksTested.size();
    const size_t numThreadSteps = result.threadsTested.size();
    const size_t numRepSteps = result.repsTested.size();

    // NaN marks configurations skipped by early termination, distinguishing them from genuine 0 GiB/s measurements
    const double UNMEASURED = std::numeric_limits<double>::quiet_NaN();

    result.bandwidth3D.assign(numBlockSteps, std::vector<std::vector<double>>(
        numThreadSteps, std::vector<double>(numRepSteps, UNMEASURED)));

    // Lowest thread count worth measuring (index into threadsTested)
    size_t lowestThreadIndex = 0;

    for (size_t bi = numBlockSteps; bi-- > 0; )
    {
        const uint32_t numBlocks = result.blocksTested[bi];
        double maxBandwidthThisBlock = 0.0;

        for (size_t ti = numThreadSteps; ti-- > lowestThreadIndex; )
        {
            const uint32_t numThreads = result.threadsTested[ti];
            double bestThisThread = 0.0;

            for (size_t ri = 0; ri < numRepSteps; ++ri)
            {
                const size_t reps = result.repsTested[ri];
                double bestThisCell = 0.0;

                for (uint32_t loadsPerBatch = 1; loadsPerBatch <= maxLoadsPerBatch; loadsPerBatch *= 2)
                {
                    auto [timeS, bandwidth] = measure(numBlocks, numThreads, loadsPerBatch, reps);

                    bestThisCell = std::max(bestThisCell, bandwidth);

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numBlocks = numBlocks;
                        result.numReps = reps;
                        result.loadsPerBatch = maxLoadsPerBatch > 1 ? loadsPerBatch : 0;
                    }
                }

                result.bandwidth3D[bi][ti][ri] = bestThisCell;
                bestThisThread = std::max(bestThisThread, bestThisCell);
            }

            if (maxBandwidthThisBlock > 0.0 && bestThisThread <= BANDWIDTH_EARLY_TERMINATION_FACTOR * maxBandwidthThisBlock)
            {
                lowestThreadIndex = ti + 1;
                break;
            }

            maxBandwidthThisBlock = std::max(maxBandwidthThisBlock, bestThisThread);
        }
    }

    return result;
}

// The repetitions swept by the cache bandwidth benchmarks (main memory uses a single pass instead)
inline std::vector<size_t> cacheBandwidthReps()
{
    std::vector<size_t> reps;
    for (size_t r = MIN_REPS; r <= MAX_REPS; r *= 2) { reps.push_back(r); }
    return reps;
}
