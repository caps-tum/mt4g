#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 512;
static constexpr auto MS_PER_SECOND = 1000.0; // ms


// Reads the array with all blocks, bypassing L1; every rep starts at the data of the next block
template <uint32_t LoadsPerBatch>
__global__ void l2ReadBandwidthKernel(uint32_t* __restrict__ dst, const uint32v4* __restrict__ src, size_t n, size_t reps)
{
    LOADS_PER_BATCH_CHECK(LoadsPerBatch, MAX_MULTI_BLOCK_LOADS_PER_BATCH);

    // read once, the "memory" clobber of the asm would reload them in every iteration
    const size_t blockStride = blockDim.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockStride;

    uint32_t dummy = 0;

    for (size_t j = 0; j < reps; ++j)
    {
        const size_t first = ((blockIdx.x + j) * blockStride + threadIdx.x) % gridStride;
        dummy ^= gridReadPass<LoadsPerBatch, true>(src, first, n, gridStride);
    }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> l2ReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numBlocks, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const size_t numElements = arraySizeBytes / sizeof(uint32v4);
    const auto kernel = withLoadsPerBatch<MAX_MULTI_BLOCK_LOADS_PER_BATCH>(loadsPerBatch, [](auto n) { return l2ReadBandwidthKernel<decltype(n)::value>; });

    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(numElements);
    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);

    // warm up L2
    kernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, numElements, WARMUP_REPS);
    util::hipCheck(hipGetLastError());

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    kernel<<<numBlocks, numThreads>>>(d_dstArr, d_srcArr, numElements, reps);
    util::hipCheck(hipGetLastError());
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    const double timeS = elapsedMs / MS_PER_SECOND;

    return {timeS, dataGiB / timeS};
}


namespace benchmark {
    CacheBandwidthResult measureL2ReadBandwidthSweep(size_t l2SizeBytes)
    {
        util::hipDeviceReset();

        const size_t arraySizeBytes = l2SizeBytes * 0.8; // 80% L2 size, L1 is bypassed

        return sweepMultiBlockBandwidth(arraySizeBytes, cacheBandwidthReps(), [&](uint32_t numBlocks, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
            return l2ReadBandwidthLauncher(arraySizeBytes, numBlocks, numThreads, loadsPerBatch, reps);
        });
    }
}
