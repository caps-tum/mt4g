#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 512;


// Writes the array with all blocks; every rep starts at the data of the next block
__global__ void l2WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n, size_t reps)
{
    // read once, on AMD the "memory" clobber of the asm would reload them in every iteration
    const size_t blockStride = blockDim.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockStride;
    const uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockStride + threadIdx.x);
    const uint32v4 data = {tid, tid + 1, tid + 2, tid + 3};

    for (size_t j = 0; j < reps; ++j)
    {
        const size_t first = ((blockIdx.x + j) * blockStride + threadIdx.x) % gridStride;

        for (size_t i = first; i < n; i += gridStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile ("flat_store_dwordx4 %0, %1\n\t" : : "v"(dst + i), "v"(data) : "memory");
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            __asm__ volatile ("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n\t" : : "l"(dst + i), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w) : "memory");
            #endif
        }
    }
}

static std::tuple<double, double> l2WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);

    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(numElements);

    // warm up L2
    l2WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, numElements, WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l2WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, numElements, reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_dstArr));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

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
