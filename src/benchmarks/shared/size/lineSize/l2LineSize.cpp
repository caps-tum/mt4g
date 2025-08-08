#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"

static constexpr auto MIN_EXPECTED_SIZE = 1 * MiB;// 1 * MiB
static constexpr auto MEASURE_SIZE = 2048;// Loads
static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
[[maybe_unused]]static constexpr auto ROUNDS = 12;// Rounds

__global__ void l2LineSizeKernel(uint32_t *pChaseArray, uint32_t *timingResults, size_t steps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // s_timings[0] is undefined, as we use it to prevent compiler optimizations / latency hiding
    __shared__ uint64_t s_timings[MEASURE_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    
    uint32_t index = 0;
    uint32_t sum = 0;
    size_t measureLength = util::min(steps, MEASURE_SIZE / sizeof(uint32_t));

    // First Round to (hopefully) fill L2 Cache while ignoring L1, GLC=1 / .cg hint
    for (uint32_t i = 0; i < steps; ++i) {
        index = __allowL1Read(pChaseArray, index);
        sum += index;
    }

    // Second Round to (hopefully) load Data from L2, GLC=1 & / .cg hint
    #ifdef __HIP_PLATFORM_NVIDIA__ // Prepare &s_timings[0] to be used in PTX to avoid latency hiding. "smem_ptr64" will contain PTX friendly address 
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t" 
        :
        : "l"(s_timings) // __shared__ uint32_t*
    );
    steps = measureLength * ROUNDS;
    #endif
    for (uint32_t i = 0; i < steps; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
        uint32_t *addr = pChaseArray + index;
        uint64_t start, end;

        asm volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t" // start = clock();

            "flat_load_dword %1, %3 " GLC "\n\t" // index = *addr;

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %2\n\t" // end = clock();

            // Last syncs
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"

            : "+s"(start) //uint64_t
            , "+v"(index) //uint32_t
            , "+s"(end) //uint64_t
            : "s"(addr) //uint32_t*
            : "memory", "scc"
        );
        #endif 
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint32_t end, start;
        uint32_t *addr = pChaseArray + index;

        asm volatile (
            "mov.u32 %0, %%clock;\n\t" // start = clock()
            "ld.global.cg.u32 %1, [%3];\n\t" // index = *addr
            // smem_ptr64 = PTX compatible address &s_timings[0], duration of this load does 
            // not matter here, as the change point will still occur
            "st.shared.u32 [smem_ptr64], %1;" 
            "mov.u32 %2, %%clock;\n\t" // end = clock()
            : "=r"(start) // uint32_t
            , "+r"(index) // uint32_t
            , "=r"(end) // uint32_t
            : "l"(addr) // uint32_t*
            : "memory"
        );
        #endif

        s_timings[i % measureLength] += end - start;
    }
    

    for (uint32_t k = 0; k < measureLength; k++) {
        #ifdef __HIP_PLATFORM_AMD__
        timingResults[k] = s_timings[k]; 
        #endif
        #ifdef __HIP_PLATFORM_NVIDIA__
        timingResults[k] = s_timings[k] / ROUNDS; 
        #endif
    }

    timingResults[0] += s_timings[0] >> util::min(sum, 32);
}


std::vector<uint32_t> l2LineSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    if (arraySizeBytes <= MIN_EXPECTED_SIZE) return {};
    util::hipDeviceReset();

    //std::cout << arraySizeBytes << " " << strideBytes << std::endl;

    size_t arraySize = arraySizeBytes / sizeof(uint32_t);
    size_t stride = strideBytes / sizeof(uint32_t);
    size_t steps = arraySize / stride;
    size_t resultBufferLength = util::min(steps, MEASURE_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    
    util::hipCheck(hipDeviceSynchronize());
    l2LineSizeKernel<<<1, util::getMaxThreadsPerBlock(), 0, util::createStreamForCU(0)>>>(d_pChaseArray, d_timingResultBuffer, steps);

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    
    timingResultBuffer.erase(timingResultBuffer.begin());
    //std::cout << "ARR SIZ " << arraySizeBytes << " : " <<  timingResultBuffer[0] << std::endl;
    return timingResultBuffer;
}



namespace benchmark {
    CacheLineSizeResult measureL2LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes) {
        std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;

        size_t measureResolution = cacheFetchGranularityBytes / CACHE_LINE_SIZE_RESOLUTION_DIVISOR; // Measure with increased accuracy
        
        for (size_t currentFetchGranularityBytes = measureResolution; currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE + measureResolution; currentFetchGranularityBytes += measureResolution) {
            std::map<size_t, std::vector<uint32_t>> t;
            for (size_t currentCacheSize = 2 * cacheSizeBytes / 3; currentCacheSize < cacheSizeBytes + cacheSizeBytes / 3; currentCacheSize += cacheSizeBytes / measureResolution) {
                timings[currentFetchGranularityBytes][currentCacheSize] = l2LineSizeLauncher(currentCacheSize, currentFetchGranularityBytes);
            }
        }
            
        auto [changePoint, confidence] = util::detectLineSizeChangePoint(timings);

        CacheLineSizeResult result = {
            timings,
            util::closestPowerOfTwo(changePoint - (changePoint % cacheFetchGranularityBytes)), // Ensure that the change point is a multiple of the fetch granularity and power of two
            confidence,
            PCHASE,
            BYTE,
            false
        };

        return result;
    }
}