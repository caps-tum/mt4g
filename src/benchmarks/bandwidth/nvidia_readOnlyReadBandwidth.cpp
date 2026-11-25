#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 128;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH == 0 && WARMUP_REPS % MAX_LOADS_PER_BATCH == 0, "loads per batch must divide the repetitions");


#ifdef __HIP_PLATFORM_NVIDIA__
// Like the NVIDIA L1 batch, but through the read-only path (ld.global.nc).
// All four words are used and all addresses differ, otherwise the compiler narrows or merges the loads.
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ uint32_t nvidiaReadOnlyLoadBatch(const uint32v4 *src)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        uint32v4 loaded;
        asm volatile (
            "ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
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
__global__ void readOnlyReadBandwidthKernel(uint32_t* __restrict__ dst, [[maybe_unused]] const uint32v4* __restrict__ src, uint64_t* __restrict__ timingResult, size_t numElements, size_t reps)
{
    LOADS_PER_BATCH_CHECK(LoadsPerBatch, MAX_LOADS_PER_BATCH);
    const uint32_t blockStride = blockDim.x;

    uint32_t dummy = 0;
    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (size_t i = threadIdx.x; i < numElements; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_NVIDIA__
            dummy ^= nvidiaReadOnlyLoadBatch<LoadsPerBatch>(src + i);
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> readOnlyReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);
    const auto kernel = withLoadsPerBatch(loadsPerBatch, [](auto n) { return readOnlyReadBandwidthKernel<decltype(n)::value>; });

    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>((arraySizeBytes + BATCH_PADDING_BYTES) / sizeof(uint32v4));
    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, 0, [&](size_t r) {
        kernel<<<1, numThreads>>>(d_dstArr, d_srcArr, d_timingResult, numElements, r);
    });

    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    namespace nvidia
    {
        CacheBandwidthResult measureReadOnlyReadBandwidthSweep(size_t arraySizeBytes)
        {
            return sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
                return readOnlyReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps);
            });
        }
    }
}
