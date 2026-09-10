#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <cstddef>
#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// DEFAULT_SAMPLE_SIZE loads should suffice to rule out random flukes

__global__ void textureLatencyKernel([[maybe_unused]]hipTextureObject_t tex, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Warm-up
    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        index = tex1Dfetch<uint32_t>(tex, index);
        #endif
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint64_t start = __timer();
        index = tex1Dfetch<uint32_t>(tex, index);
        uint64_t end = __timer();

        s_timings[k] = end - start;
        #endif
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += index >> util::min(index / 2, 32);
}

std::vector<uint32_t> textureLatencyLauncher(size_t arraySizeBytes, size_t strideBytes) { 
    util::hipDeviceReset(); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    hipTextureObject_t tex = util::createTextureObject(d_pChaseArray, arraySizeBytes);
    
    util::hipCheck(hipDeviceSynchronize());
    textureLatencyKernel<<<1, 1>>>(tex, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());

    util::hipDeviceReset();
    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheLatencyResult measureTextureLatency() {
            auto timings = textureLatencyLauncher(SAMPLE_SIZE * sizeof(uint32_t), sizeof(uint32_t));

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