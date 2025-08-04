#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// DEFAULT_SAMPLE_SIZE loads should suffice to rule out random flukes

__global__ void sharedMemoryLatencyKernel(uint32_t *timingResults) {
    uint32_t index = 0;

    __shared__ uint32_t s_pChaseArray[SAMPLE_SIZE];
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // initialize chase array
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        s_pChaseArray[i] = (i + 1) % SAMPLE_SIZE;
    }

    // prepare sink shared-space address (points to s_timings[0]) for NVIDIA
    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t s_MemSinkAddr;
    asm volatile(
        "cvta.to.shared.u64 %0, %1;\n\t" // generic ptr -> convert to shared-space
        : "=l"(s_MemSinkAddr) // uint64_t
        : "l"(s_timings) // __shared__ uint64_t*
    );
    #endif

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
        uint64_t start = __timer();
        index = s_pChaseArray[index]; // Works fine on AMD
        uint64_t end = __timer();

        s_timings[i] = end - start;
        #endif

        #ifdef __HIP_PLATFORM_NVIDIA__
        uint32_t start, end;
        uint32_t *addr = s_pChaseArray + index; // generic shared pointer

        // start timing, do shared load with pointer chase, sink-store to enforce completion, then end timing
        asm volatile (
            "{\n\t"
            ".reg .u64 smaddr;\n\t"
            "cvta.to.shared.u64 smaddr, %3;\n\t" // convert current p chase pointer to shared-space address
            "mov.u32 %0, %%clock;\n\t" // start = clock()
            "ld.shared.u32 %1, [smaddr];\n\t" // load from shared memory
            "st.shared.u32 [%4], %1;\n\t" // sink
            "mov.u32 %2, %%clock;\n\t" // end = clock()
            "}\n\t"
            : "=r"(start) // uint32_t
            , "=r"(index) // uint32_t
            , "=r"(end) // uint32_t
            : "l"(addr) // uint32_t*
            , "l"(s_MemSinkAddr) // uint64_t*
            : "memory"
        );
        s_timings[i] = end - start;
        #endif
    }

    // export results (skip first except aggregate)
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }
}



std::vector<uint32_t> sharedMemoryLatencyLauncher(/*size_t arraySizeBytes, size_t strideBytes*/) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    
    util::hipCheck(hipDeviceSynchronize());
    sharedMemoryLatencyKernel<<<1, 1>>>(d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());
    
    util::printVector(timingResultBuffer);
    return timingResultBuffer;
}

namespace benchmark {
    CacheLatencyResult measureSharedMemoryLatency() {

        auto timings = sharedMemoryLatencyLauncher();

        CacheLatencyResult result {
            timings,
            util::average(timings),
            util::percentile(timings, 0.5),
            util::percentile(timings, 0.95),
            util::stddev(timings),
            timings.size(),
            SAMPLE_SIZE,
            CYCLE,
            PCHASE
        };

        return result;
    }
}