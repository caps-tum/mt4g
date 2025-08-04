#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes

//__attribute__((optimize("O0"), noinline))
__global__ void readOnlyLineSizeKernel(const uint32_t* __restrict__ pChaseArray, uint32_t *timingResults, size_t length) {
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[MIN_EXPECTED_SIZE / sizeof(uint32_t)];

    size_t measureLength = util::min(length, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t start, end;
    uint32_t index = 0;

    for (uint32_t k = 0; k < measureLength; k++) {
        s_index[k] = 0;
        s_timings[k] = 0;
    }

    // First round
    for (uint32_t k = 0; k < length; k++) {
        index = __ldg(&pChaseArray[index]);
    }

    // Second round
    for (uint32_t k = 0; k < measureLength; k++) {
        start = clock();
        index = __ldg(&pChaseArray[index]);
        s_index[k] = index;
        end = clock();
        s_timings[k] = end - start;
    }

    for (uint32_t k = 0; k < measureLength; k++) {
        s_index[0] += s_index[k];
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}

std::vector<uint32_t> readOnlyLineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipCheck(hipDeviceReset());

    size_t arraySize = arraySizeBytes / sizeof(uint32_t);
    size_t stride = strideBytes / sizeof(uint32_t);
    size_t steps = arraySize / stride + (arraySize % stride != 0 ? 1 : 0);
    size_t resultBufferLength = std::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    

    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    

    util::hipCheck(hipDeviceSynchronize());
    readOnlyLineSizeKernel<<<1, 1>>>(d_pChaseArray, d_timingResultBuffer, steps);
    util::hipCheck(hipDeviceSynchronize());

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);


    util::hipCheck(hipDeviceReset());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureReadOnlyLineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) { 
            std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

            size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
            
            for (size_t currentFetchGranularityBytes = cacheFetchGranularityBytes; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += measureResolution) {
                for (size_t currentCacheSize = cacheSizeBytes / 2; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 2; currentCacheSize += cacheSizeBytes / measureResolution) {
                    timings[currentFetchGranularityBytes][currentCacheSize] = readOnlyLineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
                }
            }

            
        for (auto& [size, map] : timings) {
            util::pipeMapToPython(map, "RO " + std::to_string(size));
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