#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <cstddef>
#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// Loads


// Measure the constant L1.5 latency by evicting the constant L1 before every
// timed load. This ensures that each access of thread 0 misses in the constant
// L1 and is served from the L1.5 cache.
__global__ void constantL15LatencyKernel(uint32_t *timingResults, uint32_t *timingDummy, uint32_t steps, uint32_t stride) {
    uint32_t start, end;
    uint32_t indexMeasure = 0;
    uint32_t indexEvict = CONST_ARRAY_SIZE - steps * stride;

    __shared__ uint64_t s_timingResults[SAMPLE_SIZE];

    size_t measureLength = util::min(steps, SAMPLE_SIZE);

    for (uint32_t k = 0; k < measureLength; ++k) {
        if (threadIdx.x == 1) {
            // Evict the constant L1 by reading one cache worth of data
            for (uint32_t j = 0; j < steps; ++j) {
                indexEvict = arr16384AscStride0[indexEvict] + stride;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            start = clock();
            indexMeasure = arr16384AscStride0[indexMeasure] + stride;
            end = clock();
            s_timingResults[k] = end - start;
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < measureLength; ++k) {
            timingResults[k] = s_timingResults[k];
        }

        timingResults[0] += indexMeasure >> util::min(steps, 32);
    }

    if (threadIdx.x == 1) {
        timingDummy[0] = indexEvict >> util::min(steps, 32);
    }
}

std::vector<uint32_t> constantL15LatencyLauncher(size_t arraySizeBytes, size_t strideBytes) {
    util::hipCheck(hipDeviceReset());

    size_t resultBufferLength = util::min(arraySizeBytes / strideBytes, SAMPLE_SIZE);

    // Initialize device Arrays
    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingDummy = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    constantL15LatencyKernel<<<1, 2>>>(d_timingResults, d_timingDummy, arraySizeBytes / strideBytes, strideBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);

    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheLatencyResult measureConstantL15Latency(size_t constantL1SizeBytes, size_t constantL1FetchGranularityBytes) {
            std::vector<uint32_t> timings = constantL15LatencyLauncher(constantL1SizeBytes, constantL1FetchGranularityBytes);

            CacheLatencyResult result {
                timings,
                util::average(timings),
                util::percentile(timings, 0.5),
                util::percentile(timings, 0.95),
                util::stddev(timings),
                timings.size(),
                SAMPLE_SIZE,
                CYCLE,
                PCHASE
            };

            return result;
        }
    }
}