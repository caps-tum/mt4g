#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 128;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH == 0 && WARMUP_REPS % MAX_LOADS_PER_BATCH == 0, "loads per batch must divide the repetitions");


#ifdef __HIP_PLATFORM_AMD__
#define AMD_L1_LOAD(offset) "flat_load_dwordx4 v[20:23], %0 offset:" #offset "\n\t"

// All loads write v20-v23 (the data is not used)
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ void amdL1LoadBatch(const uint32v4 *src)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(src);
    AMD_LOAD_BATCH(LoadsPerBatch, AMD_L1_LOAD, "s_waitcnt vmcnt(0)\n\t", : : "v"(addr) : "memory", "v20", "v21", "v22", "v23")
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
    LOADS_PER_BATCH_CHECK(LoadsPerBatch, MAX_LOADS_PER_BATCH);
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    uint32_t dummy = 0;
    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (size_t i = threadIdx.x; i < numElements; i += blockStride)
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


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> l1ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps, hipStream_t stream)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);
    const auto kernel = withLoadsPerBatch(loadsPerBatch, [](auto n) { return l1ReadBandwidthKernel<decltype(n)::value>; });

    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>((arraySizeBytes + BATCH_PADDING_BYTES) / sizeof(uint32v4));
    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, stream, [&](size_t r) {
        kernel<<<1, numThreads, 0, stream>>>(d_dstArr, d_srcArr, d_timingResult, numElements, r);
    });

    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes)
    {
        // A single block, pinned to CU 0 on AMD
        hipStream_t stream = util::createStreamForCU(0);

        CacheBandwidthResult result = sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
            return l1ReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps, stream);
        });

        util::hipCheck(hipStreamDestroy(stream));

        return result;
    }
}
