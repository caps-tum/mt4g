#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>

static constexpr auto SAMPLE_SIZE = 2048;// 2048 Loads should suffice to rule out random flukes

__global__ void mainMemoryLatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Do not load from caches
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        uint64_t start = __timer();
        index = __forceBypassAllCacheReads(pChaseArray, index); 
        uint64_t end = __timer(); 

        s_timings[i] = end - start;
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }

    timingResults[0] += index >> util::min(index / 2, 32);
}



std::vector<uint32_t> mainMemoryLatencyLauncher(size_t arraySizeBytes, size_t strideBytes) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);
    
    util::hipCheck(hipDeviceSynchronize());
    mainMemoryLatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);
    
    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    CacheLatencyResult measureMainMemoryLatency() {

        auto timings = mainMemoryLatencyLauncher(1 * GiB, 1 * KiB);

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