#include <cstddef>
#include <hip/hip_runtime.h>

#include "const/constArray16384.hpp"
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void constantL1MissPenaltyKernel(uint32_t *timingResults, size_t steps, size_t stride) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    size_t measureLength = util::min(steps, SAMPLE_SIZE);

    // Evict the constant L1 by loading twice the cache size
    for (uint32_t k = 0; k < steps * 2; k++) {
        index = arr16384AscStride0[index] + stride;
    }

    index &= 1;
    for (uint32_t k = 0; k < measureLength; k++) {
        uint64_t start = __timer();
        index = arr16384AscStride0[index] + stride;
        uint64_t end = __timer();
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += index >> util::min(steps, 32);
}

std::vector<uint32_t> constantL1MissPenaltyLauncher(size_t constantL1CacheSizeBytes, size_t constantL1CacheLineSizeBytes) {
    util::hipDeviceReset();

    size_t steps = constantL1CacheSizeBytes / constantL1CacheLineSizeBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    // Initialize device Arrays
    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    constantL1MissPenaltyKernel<<<1, 1>>>(d_timingResults, steps, constantL1CacheLineSizeBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);

    return timingResultBuffer;
}


namespace benchmark {
    namespace nvidia {
        double measureConstantL1MissPenalty(size_t constantL1CacheSizeBytes, size_t constantL1CacheLineSizeBytes, double constantL1Latency) {
            auto timings = constantL1MissPenaltyLauncher(constantL1CacheSizeBytes, constantL1CacheLineSizeBytes);

            return std::abs(util::average(timings) - constantL1Latency);
        }
    }
}