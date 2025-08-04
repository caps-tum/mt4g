#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"
#include "const/constArray16384.hpp"

#include <hip/hip_runtime.h>
#include <optional>
#include <vector>
#include <map>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 196;// B, 
static constexpr auto SAMPLE_SIZE = 64;// Tries


__global__ void constantL1FetchGranularityKernel(uint32_t *timingResults, size_t stride) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[SAMPLE_SIZE];

    uint32_t start, end;
    uint32_t index = 0;

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        s_index[k] = 0;
        s_timings[k] = 0;
    }

    // Measure cold cache misses 
    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        start = clock();
        index = arr16384AscStride0[index] + stride;
        end = clock();
        s_index[k] = index;
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        s_index[0] += s_index[k];
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}


std::vector<uint32_t> constantL1FetchGranularityLauncher(size_t fetchGranularityToTestBytes) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    constantL1FetchGranularityKernel<<<1, 1>>>(d_timingResults, fetchGranularityToTestBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);
    
    util::hipCheck(hipDeviceReset()); 
    return timingResultBuffer;
}


namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureConstantL1FetchGranularity() {
            std::map<size_t, std::vector<uint32_t>> timings;
            
            for (size_t currentFetchGranularityBytes = sizeof(uint32_t); currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += sizeof(uint32_t)) {
                timings[currentFetchGranularityBytes] = constantL1FetchGranularityLauncher(currentFetchGranularityBytes);
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