#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <cstddef>
#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// DEFAULT_SAMPLE_SIZE loads should suffice to rule out random flukes

__global__ void constantL1LatencyKernel(uint32_t *timingResults, uint32_t stride) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Warm-up
    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        index = arr16384AscStride0[index] + stride;
    }

    index -= 257; // Null index (hopefully the compiler doesnt notice)

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        uint64_t start = __timer();
        index = arr16384AscStride0[index] + stride;
        uint64_t end = __timer();

        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += index >> util::min(index / 2, 32);
}

std::vector<uint32_t> constantL1LatencyLauncher(size_t strideBytes) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    constantL1LatencyKernel<<<1, 1>>>(d_timingResults, strideBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());

    util::hipCheck(hipDeviceReset());
    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheLatencyResult measureConstantL1Latency() {
            auto timings = constantL1LatencyLauncher(sizeof(uint32_t));

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