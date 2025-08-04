#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"

#include <hip/hip_runtime.h>
#include <optional>
#include <vector>
#include <map>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
static constexpr auto SAMPLE_SIZE = 64;// Tries

__global__ void l3FetchGranularityKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    uint32_t index = 0;

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        volatile uint64_t start = __timer();
        index = __l3Read(pChaseArray, index);
        volatile uint64_t end = __timer();
        s_timings[i] = end - start + (index & 0x1);
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }
}

std::vector<uint32_t> l3FetchGranularityLauncher(size_t arraySizeBytes, size_t fetchGranularityToTestBytes) {
    util::hipCheck(hipDeviceReset());

    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, fetchGranularityToTestBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    l3FetchGranularityKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    util::hipCheck(hipDeviceReset());
    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        CacheSizeResult measureL3FetchGranularity() {
            std::map<size_t, std::vector<uint32_t>> timings;

            for (size_t currentFetchGranularityBytes = sizeof(uint32_t); currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += sizeof(uint32_t)) {
                timings[currentFetchGranularityBytes] = l3FetchGranularityLauncher(currentFetchGranularityBytes * SAMPLE_SIZE, currentFetchGranularityBytes);
            }

            auto [changePoint, confidence] = util::detectFetchGranularityChangePoint(timings);

            CacheSizeResult result = {
                timings,
                changePoint,
                confidence,
                PCHASE,
                BYTE,
                false
            };

            return result;
        }
    }
}

