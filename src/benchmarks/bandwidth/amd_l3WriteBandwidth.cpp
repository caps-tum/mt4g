#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 512;


// Writes the array with all blocks; every rep starts at the data of the next block
__global__ void l3WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n, size_t reps)
{
    // read once, the "memory" clobber of the asm would reload them in every iteration
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
        }
    }
}

static std::tuple<double, double> l3WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, size_t reps, util::AllocatorType allocType)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);

    uint32v4 *d_dstArr = util::allocateMemory<uint32v4>(numElements, allocType);

    // warm up
    l3WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, numElements, WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l3WriteBandwidthKernel<<<numBlocks, numThreads>>>(d_dstArr, numElements, reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::freeMemory(d_dstArr, allocType);

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

    return {timeS, dataGiB / timeS};
}


namespace benchmark {
    namespace amd {
        CacheBandwidthResult measureL3WriteBandwidthSweep(size_t l2SizeBytes, size_t l3SizeBytes, util::AllocatorType allocType)
        {
            util::hipDeviceReset();

            size_t arraySizeBytes = util::max(l2SizeBytes * (util::getNumComputeDies() + 2), l3SizeBytes / 4);

            return sweepMultiBlockBandwidth(arraySizeBytes, cacheBandwidthReps(), [&](uint32_t numBlocks, uint32_t numThreads, uint32_t, size_t reps) {
                return l3WriteBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, reps, allocType);
            }, 1);
        }
    }
}
