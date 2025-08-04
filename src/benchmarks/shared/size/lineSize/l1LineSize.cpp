#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B

//__attribute__((optimize("O0"), noinline))
__global__ void l1LineSizeKernel(uint32_t *pChaseArray, uint32_t *timingResults, size_t steps) {
    // s_timings[0] is undefined, as we use it to prevent compiler optimizations / latency hiding
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    
    uint32_t index = 0;
    size_t measureLength = util::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    // First Round to (hopefully) fill L1 Cache, GLC=0 / .ca hint
    for (uint32_t i = 0; i < steps; ++i) {
        index = __allowL1Read(pChaseArray, index);
    }

    // Second Round to (hopefully) load Data from vL1d, GLC=0
    #ifdef __HIP_PLATFORM_NVIDIA__ // Prepare &s_timings[0] to be used in PTX to avoid latency hiding. "smem_ptr64" will contain PTX friendly address 
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t" 
        :
        : "l"(s_timings) // __shared__ uint32_t*
    );
    #endif
    for (uint32_t i = 0; i < measureLength; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
        uint32_t *addr = pChaseArray + index;
        uint64_t start, end;

        asm volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t" // start = clock();

            "flat_load_dword %1, %3\n\t" // index = *addr;

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %2\n\t" // end = clock();

            // Last syncs
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"

            : "+s"(start) //uint64_t
            , "+v"(index) //uint32_t
            , "+s"(end) //uint64_t
            , "+v"(addr) //uint32_t*
            :
            : "memory"
        );
        #endif 
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint32_t end, start;
        uint32_t *addr = pChaseArray + index;

        asm volatile (
            "mov.u32 %0, %%clock;\n\t" // start = clock()
            "ld.global.ca.u32 %1, [%3];\n\t" // index = *addr
            // smem_ptr64 = PTX compatible address &s_timings[0], duration of this load does 
            // not matter here, as the change point will still occur
            "st.shared.u32 [smem_ptr64], %1;" 
            "mov.u32 %2, %%clock;\n\t" // end = clock()
            : "=r"(start) // uint32_t
            , "=r"(index) // uint32_t
            , "=r"(end) // uint32_t
            : "l"(addr) // uint32_t*
            : "memory"
        );
        #endif
        s_timings[i] = end - start;
    }

    for (uint32_t k = 1; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += s_timings[0] >> util::min(steps, 32);
}

std::vector<uint32_t> l1LineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipCheck(hipDeviceReset());

    size_t stridedLength = arraySizeBytes / strideBytes;
    size_t resultBufferLength = std::min(stridedLength, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    
    util::hipCheck(hipDeviceSynchronize());
    l1LineSizeKernel<<<1, 1>>>(d_pChaseArray, d_timingResultBuffer, stridedLength);

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    
    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    CacheSizeResult measureL1LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) { 
        std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

        size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
        
        for (size_t currentFetchGranularityBytes = cacheFetchGranularityBytes; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += measureResolution) {
            for (size_t currentCacheSize = cacheSizeBytes / 2; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 2; currentCacheSize += cacheSizeBytes / measureResolution) {
                timings[currentFetchGranularityBytes][currentCacheSize] = l1LineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
            }
        }

        for (auto& [size, map] : timings) {
            util::pipeMapToPython(map, "L1 " + std::to_string(size));
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