#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 512;


static constexpr auto MS_PER_SECOND = 1000.0;// ms

__global__ void l2WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n, size_t reps) {
    // read once, the "memory" clobber of the asm would reload them in every iteration
    const uint32_t blockStride = blockDim.x;
    const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockStride;
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockStride + threadIdx.x);

    uint32v4 dummy = { tid, tid + 1, tid + 2, tid + 3 }; 

    for (size_t j = 0; j < reps; ++j) {
        tid = (((blockIdx.x + j) * blockStride) + threadIdx.x) % stride;
        
        for (size_t i = tid; i < n; i += stride) {

            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(dst + i) // uint32v4*
                , "r"(dummy.x) // int
                , "r"(dummy.y) // int
                , "r"(dummy.z) // int
                , "r"(dummy.w) // int
            );
            #endif

            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_store_dwordx4 %0, %1\n"
                :
                : "v"(dst + i) // uint32v4*
                , "v"(dummy) // uint32v4
                : "memory"
            );
            #endif

        }
    }
}


static std::tuple<double, double> l2WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps) 
{ 
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));

    // warm up L2
    l2WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), WARMUP_REPS);
    util::hipCheck(hipGetLastError());
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l2WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4), reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));
    
    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB); // Convert to GiB
    double timeS = elapsedMs / MS_PER_SECOND;
    
    return {timeS, dataGiB / timeS};
}

namespace benchmark {
    CacheBandwidthResult measureL2WriteBandwidthSweep(size_t l2SizeBytes) 
    {
        util::hipDeviceReset();

        const size_t arraySizeBytes = l2SizeBytes * 0.8; // 80% L2 size, L1 is bypassed

        return sweepMultiBlockBandwidth(arraySizeBytes, cacheBandwidthReps(), [&](uint32_t numBlocks, uint32_t numThreads, uint32_t, size_t reps) {
            return l2WriteBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps);
        }, 1);
    }
}
