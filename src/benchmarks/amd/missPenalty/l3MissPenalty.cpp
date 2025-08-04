#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cmath>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void l3MissPenaltyKernel(uint32_t *pChaseArray, uint32_t *timingResults, size_t steps) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];
    size_t measureLength = util::min(steps, SAMPLE_SIZE);

    for (uint32_t i = 0; i < steps * 2; ++i) {
        index = __l3Read(pChaseArray, index);
    }

    for (uint32_t i = 0; i < measureLength; ++i) {
        uint64_t start = __timer();
        index = __l3Read(pChaseArray, index);
        uint64_t end = __timer();
        s_timings[i] = end - start;
    }

    for (uint32_t i = 0; i < measureLength; ++i) {
        timingResults[i] = s_timings[i];
    }

    timingResults[0] = index;
}

std::vector<uint32_t> l3MissPenaltyLauncher(size_t l3CacheSizeBytes, size_t l3FetchGranularityBytes) {
    util::hipCheck(hipDeviceReset());

    size_t steps = l3CacheSizeBytes / l3FetchGranularityBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(l3CacheSizeBytes * 2, l3FetchGranularityBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    l3MissPenaltyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults, steps);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        double measureL3MissPenalty(size_t l3CacheSizeBytes, size_t l3FetchGranularityBytes, double l3Latency) {
            auto timings = l3MissPenaltyLauncher(l3CacheSizeBytes, l3FetchGranularityBytes);
            return std::abs(util::average(timings) - l3Latency);
        }
    }
}
