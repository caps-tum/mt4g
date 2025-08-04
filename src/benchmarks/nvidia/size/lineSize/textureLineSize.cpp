#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B

__global__ void textureLineSizeKernel(hipTextureObject_t tex, uint32_t *timingResults, size_t length) {
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[MIN_EXPECTED_SIZE / sizeof(uint32_t)];

    size_t measureLength = util::min(length, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t start, end;
    uint32_t index = 0;

    for (uint32_t  k = 0; k < measureLength; k++) {
        s_index[k] = 0;
        s_timings[k] = 0;
    }

    // First round
    for (uint32_t k = 0; k < length; k++) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        index = tex1Dfetch<uint32_t>(tex, index);
        #endif
    }

    // Second round
    for (uint32_t k = 0; k < measureLength; k++) {
        start = clock();
        #ifdef __HIP_PLATFORM_NVIDIA__
        index = tex1Dfetch<uint32_t>(tex, index);
        #endif
        s_index[k] = index;
        end = clock();
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
        s_index[0] += s_index[k];
    }

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}

std::vector<uint32_t> textureLineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipCheck(hipDeviceReset());

    size_t arraySize = arraySizeBytes / sizeof(uint32_t);
    size_t stride = strideBytes / sizeof(uint32_t);
    size_t steps = arraySize / stride + (arraySize % stride != 0 ? 1 : 0);
    size_t resultBufferLength = std::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    
    hipTextureObject_t tex = util::createTextureObject(d_pChaseArray, arraySizeBytes);
    

    util::hipCheck(hipDeviceSynchronize());
    textureLineSizeKernel<<<1, 1>>>(tex, d_timingResultBuffer, steps);
    util::hipCheck(hipDeviceSynchronize());


    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    

    util::hipCheck(hipDeviceReset());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureTextureLineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) { 
            std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

            size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
            
            for (size_t currentFetchGranularityBytes = cacheFetchGranularityBytes; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += measureResolution) {
                 std::map<size_t, std::vector<uint32_t>> t;
                for (size_t currentCacheSize = cacheSizeBytes / 2; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 2; currentCacheSize += cacheSizeBytes / measureResolution) {
                    timings[currentFetchGranularityBytes][currentCacheSize] = t[currentCacheSize] = textureLineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
                }
                util::pipeMapToPython(t, std::to_string(currentFetchGranularityBytes));
            }
            
        for (auto& [size, map] : timings) {
            util::pipeMapToPython(map, "TXT " + std::to_string(size));
        }

            auto [changePoint, confidence] = util::detectLineSizeChangePoint(timings);

            CacheSizeResult result = {
                util::flattenLineSizeMeasurementsToAverage(timings),
                changePoint,
                changePoint % cacheFetchGranularityBytes == 0 ? confidence : 0,
                PCHASE,
                BYTE,
                false
            };
            
            return result;
        }
    }
}