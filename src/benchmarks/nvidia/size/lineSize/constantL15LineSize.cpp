#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_EXPECTED_LINE_SIZE = 1024;// B

//__attribute__((optimize("O0"), noinline))
__global__ void constantL15LineSizeKernel(uint32_t *timingResults, size_t length, size_t stride) {
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[MIN_EXPECTED_SIZE / sizeof(uint32_t)];

    size_t measureLength = util::min(length / stride, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t start, end;
    uint32_t index = 0;

    for (uint32_t k = 0; k < measureLength; k++) {
        s_index[k] = 0;
        s_timings[k] = 0;
    }

    // First round
    for (index = 0; index < length; index += stride) {
        index = arr16384AscStride0[index];
    }

    s_index[0] = index;

    // Second round
    for (index = 0; index < measureLength * stride; index += stride) {
        start = clock();
        index = arr16384AscStride0[index];
        end = clock();
        s_timings[index / stride] = end - start;
    }

    s_index[0] += index;

    for (uint32_t k = 0; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }


    s_index[0] += index;

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}

std::vector<uint32_t> constantL15LineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipCheck(hipDeviceReset());

    size_t resultBufferLength = 16; // util::min(arraySizeBytes / strideBytes, MIN_EXPECTED_SIZE / sizeof(uint32_t));  
    
    // Allocate GPU VMemory
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    

    util::hipCheck(hipDeviceSynchronize());
    constantL15LineSizeKernel<<<1, 1>>>(d_timingResultBuffer, arraySizeBytes / sizeof(uint32_t), strideBytes / sizeof(uint32_t));
    util::hipCheck(hipDeviceSynchronize());

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    

    util::hipCheck(hipDeviceReset());

    return { timingResultBuffer[0] }; // hacky
}


namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureConstantL15LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) { 
            std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

            size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
            
            for (size_t currentFetchGranularityBytes = cacheFetchGranularityBytes; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += measureResolution) {
                for (size_t currentCacheSize = cacheSizeBytes / 2; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 2; currentCacheSize += cacheSizeBytes / measureResolution) {
                    timings[currentFetchGranularityBytes][currentCacheSize] = constantL15LineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
                }
            }

        for (auto& [size, map] : timings) {
            util::pipeMapToPython(map, "C1.5 " + std::to_string(size));
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