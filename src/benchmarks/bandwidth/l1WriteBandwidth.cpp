#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 128;


__global__ void l1WriteBandwidthKernel(uint32v4* __restrict__ dst, uint64_t* __restrict__ timingResult, size_t numElements, size_t reps)
{
    const uint32_t tid = threadIdx.x;
    const uint32v4 data = {tid, tid + 1, tid + 2, tid + 3};
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    uint64_t start = 0;

    __syncthreads();
    if (tid == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = tid; i < numElements; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "flat_store_dwordx4 %0, %1\n\t"
                :
                : "v"(reinterpret_cast<uint64_t>(dst + i)), "v"(data)
                : "memory"
            );
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile (
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n\t"
                :
                : "l"(dst + i), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w)
                : "memory"
            );
            #endif
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile ("s_waitcnt vmcnt(0) lgkmcnt(0)\n\t" : : : "memory"); // the compiler does not track the asm stores
    #endif

    __syncthreads();
    if (tid == 0) { *timingResult = __globaltimer() - start; }
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> l1WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);

    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(numElements);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, stream, [&](size_t r) {
        l1WriteBandwidthKernel<<<1, numThreads, 0, stream>>>(d_dstArr, d_timingResult, numElements, r);
    });

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    CacheBandwidthResult measureL1WriteBandwidthSweep(size_t arraySizeBytes)
    {
        // A single block, pinned to CU 0 on AMD
        hipStream_t stream = util::createStreamForCU(0);

        CacheBandwidthResult result = sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t, size_t reps) {
            return l1WriteBandwidthLauncher(arraySizeBytes, numThreads, reps, stream);
        }, 1);

        util::hipCheck(hipStreamDestroy(stream));

        return result;
    }
}
