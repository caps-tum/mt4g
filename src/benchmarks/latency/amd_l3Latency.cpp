#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void l3LatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults, size_t steps) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    for (uint32_t i = 0; i < steps; ++i) { // Evicts L2, keeps L3
        index = __l3Read(pChaseArray, index);
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        uint64_t start = __timer();
        index = __l3Read(pChaseArray, index);
        uint64_t end = __timer();

        s_timings[i] = end - start;
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }

    timingResults[0] += index >> util::min(index / 2, 32);
}

__global__ void l3LatencyInitKernel(uint32_t *destination, const uint32_t *source, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        destination[index] = source[index];
    }
}

std::vector<uint32_t> l3LatencyLauncher(size_t arraySizeBytes, size_t strideBytes,
                                        util::AllocatorType allocType) {
    util::hipDeviceReset();

    std::vector<uint32_t> hostChaseArray = util::generatePChaseArray(arraySizeBytes, strideBytes);
    uint32_t *d_pChaseArray = nullptr;
    if (allocType == util::AllocatorType::HipMalloc) {
        d_pChaseArray = util::allocateGPUMemory(hostChaseArray);
    } else {
        d_pChaseArray = util::allocateMemory<uint32_t>(hostChaseArray.size(), allocType);

        // Initialize from the GPU before the existing L3 warm-up traversal.
        uint32_t *d_initArray = util::allocateGPUMemory(hostChaseArray);
        uint32_t threads = util::min(
            util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
        uint32_t blocks = util::getNumberOfComputeUnits()
            * util::getDeviceProperties().maxBlocksPerMultiProcessor;
        l3LatencyInitKernel<<<blocks, threads>>>(
            d_pChaseArray, d_initArray, hostChaseArray.size());
        util::hipCheck(hipDeviceSynchronize());
        util::hipCheck(hipFree(d_initArray));
    }
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    l3LatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults, arraySizeBytes / strideBytes);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());

    util::freeMemory(d_pChaseArray, allocType);
    util::hipCheck(hipFree(d_timingResults));
    util::hipDeviceReset();
    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        CacheLatencyResult measureL3Latency(size_t l2SizeBytes, size_t l2FetchGranularityBytes,
                                            util::AllocatorType allocType) {
            auto timings = l3LatencyLauncher(l2SizeBytes * 2 + SAMPLE_SIZE * l2FetchGranularityBytes,
                                             l2FetchGranularityBytes, allocType);

            CacheLatencyResult result {
                timings,
                util::average(timings),
                util::percentile(timings, 0.5),
                util::percentile(timings, 0.95),
                util::stdev(timings),
                timings.size(),
                SAMPLE_SIZE,
                CYCLE,
                PCHASE
            };

            return result;
        }
    }
}
