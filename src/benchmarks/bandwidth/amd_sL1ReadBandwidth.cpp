#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 128;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH == 0 && WARMUP_REPS % MAX_LOADS_PER_BATCH == 0, "loads per batch must divide the repetitions");


#ifdef __HIP_PLATFORM_AMD__
#define AMD_SCALAR_LOAD(offset) "s_load_dwordx16 s[40:55], %0, " #offset "\n\t"

// All loads write s40-s55 (the data is not used) and the wait is in the same asm,
// so the compiler cannot reuse these registers while the loads are in flight
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ void amdScalarLoadBatch(const uint32v16 *src)
{
    AMD_LOAD_BATCH(LoadsPerBatch, AMD_SCALAR_LOAD, "s_waitcnt lgkmcnt(0)\n\t", : : "s"(src) : "memory",
        "s40", "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52", "s53", "s54", "s55")
}
#endif


// Each wave reads every numWaves-th 64-byte element with scalar loads
template <uint32_t LoadsPerBatch>
__global__ void sL1ReadBandwidthKernel([[maybe_unused]] const uint32v16* __restrict__ src, uint64_t* __restrict__ timingResult, uint32_t waveSize, uint32_t numElements, size_t reps)
{
    LOADS_PER_BATCH_CHECK(LoadsPerBatch, MAX_LOADS_PER_BATCH);

    const uint32_t numWaves = blockDim.x / waveSize;
    uint32_t waveId = threadIdx.x / waveSize;

    #ifdef __HIP_PLATFORM_AMD__
    // unlike inline asm, the builtin marks the result wave-uniform, so the addresses stay in SGPRs as s_load requires
    waveId = __builtin_amdgcn_readfirstlane(waveId);
    #endif

    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (uint32_t i = waveId; i < numElements; i += numWaves)
        {
            #ifdef __HIP_PLATFORM_AMD__
            amdScalarLoadBatch<LoadsPerBatch>(src + i);
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sL1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const uint32_t numElements = arraySizeBytes / sizeof(uint32v16);
    const auto kernel = withLoadsPerBatch(loadsPerBatch, [](auto n) { return sL1ReadBandwidthKernel<decltype(n)::value>; });

    uint32v16 *d_srcArr = util::allocateGPUMemory<uint32v16>((arraySizeBytes + BATCH_PADDING_BYTES) / sizeof(uint32v16));
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, 0, [&](size_t r) {
        kernel<<<1, numThreads>>>(d_srcArr, d_timingResult, util::getWarpSize(), numElements, r);
    });

    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) numElements * sizeof(uint32v16) * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark {
    namespace amd {
        CacheBandwidthResult measureScalarL1ReadBandwidthSweep(size_t arraySizeBytes)
        {
            return sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
                return sL1ReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps);
            });
        }
    }
}
