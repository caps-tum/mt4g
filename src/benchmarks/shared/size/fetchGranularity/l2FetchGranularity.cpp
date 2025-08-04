#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"

#include <hip/hip_runtime.h>
#include <optional>
#include <vector>
#include <map>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
static constexpr auto SAMPLE_SIZE = 32;// Tries


__global__ void l2FetchGranularityKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    
    uint32_t index = 0;

    #ifdef __HIP_PLATFORM_NVIDIA__ // Prepare &s_timings[0] to be used in PTX to avoid latency hiding. "smem_ptr64" will contain PTX friendly address 
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t" 
        :
        : "l"(s_timings) // __shared__ uint32_t*
    );
    #endif
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
        uint64_t start = clock();
        index = __forceL1MissRead(pChaseArray, index);
        uint64_t end = clock();
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


    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }

    timingResults[0] += SAMPLE_SIZE >> util::min(index, 32);
}


std::vector<uint32_t> l2FetchGranularityLauncher(size_t arraySizeBytes, size_t fetchGranularityToTestBytes) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generateRandomizedPChaseArray(arraySizeBytes, fetchGranularityToTestBytes));

    util::hipCheck(hipDeviceSynchronize());

    l2FetchGranularityKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}


namespace benchmark {
    CacheSizeResult measureL2FetchGranularity() {
        std::map<size_t, std::vector<uint32_t>> timings;
        
        for (size_t currentFetchGranularityBytes = sizeof(uint32_t); currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += sizeof(uint32_t)) {
            timings[currentFetchGranularityBytes] = l2FetchGranularityLauncher(currentFetchGranularityBytes * SAMPLE_SIZE, currentFetchGranularityBytes);
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