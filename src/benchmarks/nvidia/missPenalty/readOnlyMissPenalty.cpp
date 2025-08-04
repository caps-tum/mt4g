#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void readOnlyMissPenaltyKernel(const uint32_t* __restrict__ pChaseArray, uint32_t *timingResults, size_t length) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    size_t measureLength = util::min(length, SAMPLE_SIZE);

    // Evict by loading four times the cache size
    for (uint32_t k = 0; k < length * 4; k++) {
        index = __ldg(&pChaseArray[index]);
    }

    // index = 0, compiler doesnt know though
    // Second round
    for (uint32_t k = 0; k < measureLength; k++) {
        uint64_t start = __timer();
        index = __ldg(&pChaseArray[index]);
        uint64_t end = __timer();
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = index;
}

std::vector<uint32_t> readOnlyMissPenaltyLauncher(size_t readOnlyCacheSizeBytes, size_t readOnlyCacheLineSizeBytes) {
    util::hipCheck(hipDeviceReset());

    size_t steps = readOnlyCacheSizeBytes / readOnlyCacheLineSizeBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    auto initializerArray = util::generatePChaseArray(readOnlyCacheSizeBytes * 4, readOnlyCacheLineSizeBytes);

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(initializerArray);

    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    readOnlyMissPenaltyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults, steps);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}


namespace benchmark {
    namespace nvidia {
        double measureReadOnlyMissPenalty(size_t readOnlyCacheSizeBytes, size_t readOnlyCacheLineSizeBytes, double readOnlyLatency) {
            auto timings = readOnlyMissPenaltyLauncher(readOnlyCacheSizeBytes, readOnlyCacheLineSizeBytes);

            return std::abs(util::average(timings) - readOnlyLatency);
        }
    }
}