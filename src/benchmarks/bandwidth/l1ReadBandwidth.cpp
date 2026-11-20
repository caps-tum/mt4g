#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

static constexpr auto WARMUP_REPS = 128;

// Loads per batch are swept as powers of two up to this, beyond what the hardware keeps in flight, to see the plateau
static constexpr uint32_t MAX_LOADS_PER_BATCH = 128;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH == 0 && WARMUP_REPS % MAX_LOADS_PER_BATCH == 0, "loads per batch must divide the repetitions");

// Load k of a batch reads k * 128 bytes further, so the loads of a batch hit different cache lines
// (on AMD k % 32, as the offset field of the instruction has 12 bits)
static constexpr size_t BATCH_STRIDE_BYTES = 128;
static constexpr size_t BATCH_OFFSET_BYTES = MAX_LOADS_PER_BATCH * BATCH_STRIDE_BYTES;


#ifdef __HIP_PLATFORM_AMD__
#define AMD_L1_LOAD(offset) "flat_load_dwordx4 v[20:23], %0 offset:" #offset "\n\t"
#define AMD_L1_LOADS_1   AMD_L1_LOAD(0)
#define AMD_L1_LOADS_2   AMD_L1_LOADS_1 AMD_L1_LOAD(128)
#define AMD_L1_LOADS_4   AMD_L1_LOADS_2 AMD_L1_LOAD(256) AMD_L1_LOAD(384)
#define AMD_L1_LOADS_8   AMD_L1_LOADS_4 AMD_L1_LOAD(512) AMD_L1_LOAD(640) AMD_L1_LOAD(768) AMD_L1_LOAD(896)
#define AMD_L1_LOADS_16  AMD_L1_LOADS_8 AMD_L1_LOAD(1024) AMD_L1_LOAD(1152) AMD_L1_LOAD(1280) AMD_L1_LOAD(1408) \
                         AMD_L1_LOAD(1536) AMD_L1_LOAD(1664) AMD_L1_LOAD(1792) AMD_L1_LOAD(1920)
#define AMD_L1_LOADS_32  AMD_L1_LOADS_16 AMD_L1_LOAD(2048) AMD_L1_LOAD(2176) AMD_L1_LOAD(2304) AMD_L1_LOAD(2432) \
                         AMD_L1_LOAD(2560) AMD_L1_LOAD(2688) AMD_L1_LOAD(2816) AMD_L1_LOAD(2944) \
                         AMD_L1_LOAD(3072) AMD_L1_LOAD(3200) AMD_L1_LOAD(3328) AMD_L1_LOAD(3456) \
                         AMD_L1_LOAD(3584) AMD_L1_LOAD(3712) AMD_L1_LOAD(3840) AMD_L1_LOAD(3968)
#define AMD_L1_LOADS_64  AMD_L1_LOADS_32 AMD_L1_LOADS_32
#define AMD_L1_LOADS_128 AMD_L1_LOADS_64 AMD_L1_LOADS_64

// All loads write v20-v23 (the data is not used) and the wait is in the same asm,
// so the compiler cannot reuse these registers while the loads are in flight
#define AMD_L1_BATCH(loads) __asm__ volatile (loads "s_waitcnt vmcnt(0)\n\t" : : "v"(addr) : "memory", "v20", "v21", "v22", "v23")

template <uint32_t LoadsPerBatch>
__device__ __forceinline__ void amdL1LoadBatch(const uint32v4 *src)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(src);

    if constexpr (LoadsPerBatch == 1)   { AMD_L1_BATCH(AMD_L1_LOADS_1); }
    if constexpr (LoadsPerBatch == 2)   { AMD_L1_BATCH(AMD_L1_LOADS_2); }
    if constexpr (LoadsPerBatch == 4)   { AMD_L1_BATCH(AMD_L1_LOADS_4); }
    if constexpr (LoadsPerBatch == 8)   { AMD_L1_BATCH(AMD_L1_LOADS_8); }
    if constexpr (LoadsPerBatch == 16)  { AMD_L1_BATCH(AMD_L1_LOADS_16); }
    if constexpr (LoadsPerBatch == 32)  { AMD_L1_BATCH(AMD_L1_LOADS_32); }
    if constexpr (LoadsPerBatch == 64)  { AMD_L1_BATCH(AMD_L1_LOADS_64); }
    if constexpr (LoadsPerBatch == 128) { AMD_L1_BATCH(AMD_L1_LOADS_128); }
}
#endif


#ifdef __HIP_PLATFORM_NVIDIA__
// The loads of a batch do not depend on each other, so they overlap; the hardware waits only when the XOR uses a result.
// All four words are used and all addresses differ, otherwise the compiler narrows or merges the loads.
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ uint32_t nvidiaL1LoadBatch(const uint32v4 *src)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        uint32v4 loaded;
        asm volatile (
            "ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(loaded.x), "=r"(loaded.y), "=r"(loaded.z), "=r"(loaded.w)
            : "l"(src + k * (BATCH_STRIDE_BYTES / sizeof(uint32v4)))
            : "memory"
        );
        dummy ^= loaded.x ^ loaded.y ^ loaded.z ^ loaded.w;
    }

    return dummy;
}
#endif


template <uint32_t LoadsPerBatch>
__global__ void l1ReadBandwidthKernel(uint32_t* __restrict__ dst, const uint32v4* __restrict__ src, uint64_t* __restrict__ timingResult, size_t numElements, size_t reps)
{
    static_assert(LoadsPerBatch <= MAX_LOADS_PER_BATCH && (LoadsPerBatch & (LoadsPerBatch - 1)) == 0, "loads per batch must be a power of two up to MAX_LOADS_PER_BATCH");

    uint32_t dummy = 0;
    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (size_t i = threadIdx.x; i < numElements; i += blockDim.x)
        {
            #ifdef __HIP_PLATFORM_AMD__
            amdL1LoadBatch<LoadsPerBatch>(src + i);
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            dummy ^= nvidiaL1LoadBatch<LoadsPerBatch>(src + i);
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


using L1ReadBandwidthKernel = void (*)(uint32_t*, const uint32v4*, uint64_t*, size_t, size_t);

static L1ReadBandwidthKernel l1ReadBandwidthKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:   return l1ReadBandwidthKernel<1>;
        case 2:   return l1ReadBandwidthKernel<2>;
        case 4:   return l1ReadBandwidthKernel<4>;
        case 8:   return l1ReadBandwidthKernel<8>;
        case 16:  return l1ReadBandwidthKernel<16>;
        case 32:  return l1ReadBandwidthKernel<32>;
        case 64:  return l1ReadBandwidthKernel<64>;
        case 128: return l1ReadBandwidthKernel<128>;
        default:  throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
    }
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> l1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps, hipStream_t stream)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);
    const L1ReadBandwidthKernel kernel = l1ReadBandwidthKernelFor(loadsPerBatch);

    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>((arraySizeBytes + BATCH_OFFSET_BYTES) / sizeof(uint32v4));
    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    uint64_t ticks = 0;

    // Warm up L1; a batch that needs more registers than this block size allows does not launch and counts as 0
    kernel<<<1, numThreads, 0, stream>>>(d_dstArr, d_srcArr, d_timingResult, numElements, WARMUP_REPS);
    const hipError_t launchError = hipGetLastError();

    if (launchError != hipErrorLaunchOutOfResources)
    {
        util::hipCheck(launchError);

        kernel<<<1, numThreads, 0, stream>>>(d_dstArr, d_srcArr, d_timingResult, numElements, reps);
        util::hipCheck(hipGetLastError());
        util::hipCheck(hipStreamSynchronize(stream));

        ticks = util::copyFromDevice(d_timingResult, 1)[0];
    }

    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    if (ticks == 0) { return {0.0, 0.0}; }

    const double timeS = (double) ticks / (util::getWallClockRateKHz() * 1000.0);
    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark
{
    CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes)
    {
        // A single block, pinned to CU 0 on AMD
        hipStream_t stream = util::createStreamForCU(0);

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

                double bestBandwidth = 0.0;

                for (uint32_t loadsPerBatch = 1; loadsPerBatch <= MAX_LOADS_PER_BATCH; loadsPerBatch *= 2)
                {
                    auto [timeS, bandwidth] = l1ReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps, stream);

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

        util::hipCheck(hipStreamDestroy(stream));

        return result;
    }
}
