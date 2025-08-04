#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"

#include <hip/hip_runtime.h>
#include <optional>
#include <vector>
#include <map>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
static constexpr auto SAMPLE_SIZE = 64;// Tries


__global__ void l1FetchGranularityKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    // s_timings[0] is undefined, as we use it to prevent compiler optimizations / latency hiding
    __shared__ uint64_t s_timings[SAMPLE_SIZE]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    
    uint32_t index = 0;

    // Test Cold Cache Misses. The larger the Cache Line Size to be tested is, the more cache misses will occur
    // After loading a new cache line, the next cacheFetchGranularityBytes / currentStrideBytes reads will be cache hits. 
    // If cacheFetchGranularityByte == currentStrideBytes every load has to be a cache miss (except for prefetching, which (hopefully)
    // does not happen for L1) 
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

    for (uint32_t k = 1; k < SAMPLE_SIZE; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += s_timings[0] >> util::min(index, 32);
}


std::vector<uint32_t> l1FetchGranularityLauncher(size_t arraySizeBytes, size_t fetchGranularityToTestBytes) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, fetchGranularityToTestBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);


    util::hipCheck(hipDeviceSynchronize());
    l1FetchGranularityKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    
    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);
    
    timingResultBuffer.erase(timingResultBuffer.begin());
    
    return timingResultBuffer;
}


namespace benchmark {
    CacheSizeResult measureL1FetchGranularity() {
        std::map<size_t, std::vector<uint32_t>> timings;
        
        for (size_t currentFetchGranularityBytes = sizeof(uint32_t); currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += sizeof(uint32_t)) {
            timings[currentFetchGranularityBytes] = l1FetchGranularityLauncher(currentFetchGranularityBytes * SAMPLE_SIZE, currentFetchGranularityBytes);
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