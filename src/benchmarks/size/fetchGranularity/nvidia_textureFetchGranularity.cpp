#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"

#include <hip/hip_runtime.h>
#include <optional>
#include <vector>
#include <map>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
static constexpr auto SAMPLE_SIZE = 64;// Tries


__global__ void textureFetchGranularityKernel([[maybe_unused]]hipTextureObject_t tex, uint32_t *timingResults) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[SAMPLE_SIZE];

    [[maybe_unused]]uint32_t start, end;
    [[maybe_unused]]uint32_t index = 0;
    
    // Provoke cold cache misses
    for (uint32_t k = 0; k < SAMPLE_SIZE; ++k) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        start = clock();
        index = tex1Dfetch<uint32_t>(tex, index);
        s_index[k] = index;
        end = clock();
        s_timings[k] = end - start;
        #endif
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        timingResults[k] = s_timings[k];
        s_index[0] += s_index[k];
    }

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}


std::vector<uint32_t> textureFetchGranularityLauncher(size_t arraySizeBytes, size_t fetchGranularityToTestBytes) { 
    util::hipDeviceReset(); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, fetchGranularityToTestBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    hipTextureObject_t tex = util::createTextureObject(d_pChaseArray, arraySizeBytes);
    
    util::hipCheck(hipDeviceSynchronize());
    textureFetchGranularityKernel<<<1, 1>>>(tex, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);
    
    
    util::hipDeviceReset(); 
    return timingResultBuffer;
}


namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureTextureFetchGranularity() {
            std::map<size_t, std::vector<uint32_t>> timings;
            
            for (size_t currentFetchGranularityBytes = sizeof(uint32_t); currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += sizeof(uint32_t)) {
                timings[currentFetchGranularityBytes] = textureFetchGranularityLauncher(currentFetchGranularityBytes * SAMPLE_SIZE, currentFetchGranularityBytes);
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