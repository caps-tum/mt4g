#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// DEFAULT_SAMPLE_SIZE loads should suffice to rule out random flukes

__global__ void l2LatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Warm-up to populate caches
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        index = __forceL1MissRead(pChaseArray, index);
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        uint64_t start = __timer();
        index = __forceL1MissRead(pChaseArray, index);
        uint64_t end = __timer();

        s_timings[i] = end - start;
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }

    timingResults[0] += index >> util::min(index / 2, 32);
}

std::vector<uint32_t> l2LatencyLauncher(size_t arraySizeBytes, size_t strideBytes) { 
    util::hipDeviceReset(); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);
    

    util::hipCheck(hipDeviceSynchronize());
    l2LatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());
    util::hipDeviceReset(); 
    return timingResultBuffer;
}

namespace benchmark {
    CacheLatencyResult measureL2Latency() {

        auto timings = l2LatencyLauncher(SAMPLE_SIZE * sizeof(uint32_t), sizeof(uint32_t));

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