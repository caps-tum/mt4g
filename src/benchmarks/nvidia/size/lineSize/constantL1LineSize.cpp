#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B

//__attribute__((optimize("O0"), noinline))
__global__ void constantL1LineSizeKernel(uint32_t *timingResults, size_t steps, uint32_t stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 

    size_t measureLength = util::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t maxIndex = steps * stride;
    uint32_t index = 0;
    uint32_t sum = 0;

    // First round
    for (uint32_t k = 0; k < steps; ++k) {
        index = (arr16384AscStride0[index] + stride) % maxIndex;
        sum += index;
    }

    index &= 1;


    // Second round
    for (uint32_t k = 0; k < measureLength; ++k) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint32_t latency;
        asm volatile(
            ".reg .u32 r_start, r_end, r_tmp;\n\t" 
            ".reg .u64 r_off, r_addr, r_base;\n\t"
            "mov.u64 r_base, arr16384AscStride0;\n\t" 
            "mul.wide.u32 r_off, %1, 4;\n\t"            
            "add.u64 r_addr, r_base, r_off;\n\t"        

            "mov.u32 r_start, %%clock;\n\t"
            "ld.const.u32 r_tmp, [r_addr];\n\t"         
            "add.u32 r_tmp, r_tmp, %2;\n\t"             
            "rem.u32 r_tmp, r_tmp, %3;\n\t"             
            "mov.u32 r_end, %%clock;\n\t"

            "sub.u32 %0, r_end, r_start;\n\t"           
            "mov.u32 %1, r_tmp;\n\t"                    
            : "=r"(latency)
            , "+r"(index)
            : "r"(stride)
            , "r"(maxIndex)
            : "memory"
        );

        s_timings[k] = latency; 
        #endif
    }


    for (uint32_t k = 1; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = sum >> util::min(index, 32);
}

std::vector<uint32_t> constantL1LineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipDeviceReset();

    size_t resultBufferLength = util::min(arraySizeBytes / strideBytes, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    constantL1LineSizeKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_timingResultBuffer, arraySizeBytes / strideBytes, strideBytes / sizeof(uint32_t));
    
    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);

    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheLineSizeResult measureConstantL1LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) {
            std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

            size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
            
            for (size_t currentFetchGranularityBytes = measureResolution; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE + measureResolution; currentFetchGranularityBytes += measureResolution) {
                for (size_t currentCacheSize = 2 * cacheSizeBytes / 3 ; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 3; currentCacheSize += cacheSizeBytes / measureResolution) {
                    timings[currentFetchGranularityBytes][currentCacheSize] = constantL1LineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
                }
            }
            
            auto [changePoint, confidence] = util::detectLineSizeChangePoint(timings);

            CacheLineSizeResult result = {
                timings,
                changePoint - (changePoint % cacheFetchGranularityBytes), // Ensure that the change point is a multiple of the fetch granularity
                confidence,
                PCHASE,
                BYTE,
                false
            };

            return result;
        }
    }
}