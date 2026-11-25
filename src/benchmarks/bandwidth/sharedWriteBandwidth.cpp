#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 128;


__global__ void sharedWriteBandwidthKernel(uint64_t* __restrict__ timingResult, uint32_t numElements, size_t reps)
{
    const uint32_t tid = threadIdx.x;
    const uint32v4 data = {tid, tid + 1, tid + 2, tid + 3};
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    extern __shared__ uint32v4 memory[];

    #ifdef __HIP_PLATFORM_AMD__
    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)); // byte address in the LDS
    #endif

    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t base;
    asm volatile ("cvta.to.shared.u64 %0, %1;" : "=l"(base) : "l"(memory));
    #endif

    uint64_t start = 0;

    __syncthreads();
    if (tid == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (uint32_t i = tid; i < numElements; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile ("ds_write_b128 %0, %1\n\t" : : "v"(base + i * 16u), "v"(data) : "memory");
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile (
                "st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n\t"
                :
                : "l"(base + i * 16u), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w)
                : "memory"
            );
            #endif
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile ("s_waitcnt lgkmcnt(0)\n\t" : : : "memory"); // the compiler does not track the asm stores
    #endif

    __syncthreads();
    if (tid == 0) { *timingResult = __globaltimer() - start; }
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sharedWriteBandwidthLauncher(uint32_t arraySizeBytes, uint32_t numThreads, size_t reps)
{
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, 0, [&](size_t r) {
        sharedWriteBandwidthKernel<<<1, numThreads, arraySizeBytes>>>(d_timingResult, arraySizeBytes / sizeof(uint32v4), r);
    });

    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    CacheBandwidthResult measureSharedWriteBandwidthSweep(uint32_t arraySizeBytes)
    {
        return sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t, size_t reps) {
            return sharedWriteBandwidthLauncher(arraySizeBytes, numThreads, reps);
        }, 1);
    }
}
