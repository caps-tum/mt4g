#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void l1MissPenaltyKernel(uint32_t* pChaseArray, uint32_t *timingResults, size_t length) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    size_t measureLength = util::min(length, SAMPLE_SIZE);

    // Load four times the cache size to evict previously loaded values
    for (uint32_t k = 0; k < length * 4; ++k) {
        index = __allowL1Read(pChaseArray, index);
    }

    // index = 0, compiler doesnt know though
    // Second round
    for (uint32_t k = 0; k < measureLength; k++) {
        uint64_t start = __timer();
        index = __allowL1Read(pChaseArray, index);
        uint64_t end = __timer();
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = index;
}

std::vector<uint32_t> l1MissPenaltyLauncher(size_t l1CacheSizeBytes, size_t l1CacheLineSizeBytes) {
    util::hipCheck(hipDeviceReset());

    size_t steps = l1CacheSizeBytes / l1CacheLineSizeBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    auto initializerArray = util::generatePChaseArray(l1CacheSizeBytes * 4, l1CacheLineSizeBytes);

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(initializerArray);
    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);


    util::hipCheck(hipDeviceSynchronize());
    l1MissPenaltyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults, steps);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);

    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}


namespace benchmark {
    double measureL1MissPenalty(size_t l1CacheSizeBytes, size_t l1CacheLineSizeBytes, double l1Latency) {
        auto timings = l1MissPenaltyLauncher(l1CacheSizeBytes, l1CacheLineSizeBytes);

        return std::abs(util::average(timings) - l1Latency);
    }
}