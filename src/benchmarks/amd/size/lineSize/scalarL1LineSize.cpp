#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B

__global__ void scalarL1LineSizeKernel(uint32_t *timingResults, size_t steps, uint32_t stride) {
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 

    size_t measureLength = util::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t index = 0;

    // First round
    for (uint32_t k = 0; k < steps; ++k) {
        index = arr16384AscStride0[index] + stride;
    }

    uint32_t sum = index;
    index = 0;

    // Second round
    for (uint32_t k = 0; k < measureLength; ++k) {
        #ifdef __HIP_PLATFORM_AMD__
        uint64_t start, end;
        uint32_t *addr = arr16384AscStride0 + index;

        asm volatile(
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t" // start = clock();

            "s_load_dword %2, %3, 0\n\t" // index = *addr;

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %1\n\t" // end = clock();

            "s_add_u32 %2, %2, %4\n\t" // index = index + stride

            // Last syncs
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"

            : "+s"(start) // uint64_t
            , "+s"(end) // uint64_t
            , "+s"(index) // uint32_t
            , "+s"(addr) // uint32_t*
            , "+s"(stride) // uint32_t
            :
            : "memory"
        );
        s_timings[k] = end - start;
        #endif

    }



    for (uint32_t k = 1; k < measureLength; ++k) { // from 1 because idx 0 is trash, to be discussed
        timingResults[k] = s_timings[k];
    }

    //timingResults[0] =  (end - start) / measureLength;
    timingResults[0] = (index + sum & 0x8) >> 2;
    //timingResults[2] =  sum;
}

std::vector<uint32_t> scalarL1LineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipCheck(hipDeviceReset());

    //std::cout << arraySizeBytes << std::endl;

    size_t resultBufferLength = util::min(arraySizeBytes / strideBytes, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1LineSizeKernel<<<1, 1>>>(d_timingResultBuffer, arraySizeBytes / strideBytes, strideBytes / sizeof(uint32_t));

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin());
    util::hipCheck(hipDeviceReset());

    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        CacheSizeResult measureScalarL1LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) { 
            std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

            size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
            
            for (size_t currentFetchGranularityBytes = cacheFetchGranularityBytes; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += measureResolution) {
                for (size_t currentCacheSize = cacheSizeBytes / 2; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 2; currentCacheSize += cacheSizeBytes / measureResolution) {
                    timings[currentFetchGranularityBytes][currentCacheSize] = scalarL1LineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
                }
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