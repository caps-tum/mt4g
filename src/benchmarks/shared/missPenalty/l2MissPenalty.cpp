#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void l2MissPenaltyKernel(uint32_t* pChaseArray, uint32_t *timingResults, size_t steps) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE];
    uint32_t index = 0;

    size_t measureLength = util::min(steps, SAMPLE_SIZE);

    // Evict L2 by loading four times the cache size with L1-bypassing reads
    for (uint32_t k = 0; k < steps * 4; ++k) {
        index = __forceL1MissRead(pChaseArray, index);
    }

    // index = 0, compiler doesnt know though
    // Second round
    for (uint32_t k = 0; k < measureLength; ++k) {
        uint64_t start = __timer();
        index = __forceL1MissRead(pChaseArray, index);
        uint64_t end = __timer();
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = index;
}

std::vector<uint32_t> l2MissPenaltyLauncher(size_t l2CacheSizeBytes, size_t l2CacheLineSizeBytes) {
    util::hipCheck(hipDeviceReset());

    size_t steps = l2CacheSizeBytes / l2CacheLineSizeBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    auto initializerArray = util::generatePChaseArray(l2CacheSizeBytes * 4, l2CacheLineSizeBytes);
    uint32_t *d_pChaseArray = util::allocateGPUMemory(initializerArray);
    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    l2MissPenaltyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults, steps);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin()); 

    
    return timingResultBuffer;
}

namespace benchmark {
    double measureL2MissPenalty(size_t l2CacheSizeBytes, size_t l2CacheLineSizeBytes, double l2Latency) {
        auto timings = l2MissPenaltyLauncher(l2CacheSizeBytes, l2CacheLineSizeBytes);
        return std::abs(util::average(timings) - l2Latency);
    }
}
