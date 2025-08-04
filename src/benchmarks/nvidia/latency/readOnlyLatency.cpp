#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <cstddef>
#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// DEFAULT_SAMPLE_SIZE loads should suffice to rule out random flukes

__global__ void readOnlyLatencyKernel(const uint32_t* __restrict__ pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // prepare shared-memory sink: use s_timings[0] as sink for enforcing load completion
    uint64_t s_MemSinkAddr;
    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        "cvta.to.shared.u64 %0, %1;\n\t"  // generic -> shared-space address
        : "=l"(s_MemSinkAddr) // uint64_t
        : "l"(&s_timings[0]) // __shared__ uint64_t*
    );
    #endif

    // warm-up to populate the read-only path (like __ldg)
    for (uint32_t k = 0; k < SAMPLE_SIZE; ++k) {
        const uint32_t* addr = pChaseArray + index;
        uint32_t tmp;
        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "ld.global.nc.u32 %0, [%1];\n\t"  // read-only cache load path
            : "=r"(tmp) // uint32_t
            : "l"(addr) // uint32_t*
            : "memory"
        );
        #endif
        index = tmp;
    }

    // measurement loop: do one dependent read-only load per iteration and sink it into shared mem
    for (uint32_t k = 0; k < SAMPLE_SIZE; ++k) {
        uint64_t start, end;
        uint32_t newIndex;
        const uint32_t* addr = pChaseArray + index;

        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "mov.u64 %0, %%clock64;\n\t" // start = clock()
            "ld.global.nc.u32 %1, [%3];\n\t" // read-only load 
            "st.shared.u32 [%4], %1;\n\t" // sink: force use of loaded value before proceeding
            "mov.u64 %2, %%clock64;\n\t" // end = clock()
            : "=l"(start) // uint64_t
            , "=r"(newIndex) // uint32_t
            , "=l"(end) // uint64_t
            , "l"(addr) // uint32_t*
            , "l"(s_MemSinkAddr) // uint64_t* (shared memory sink)
            : "memory"
        );
        #endif

        index = newIndex;
        s_timings[k] = end - start; 
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = index; // dead code elimination prevention
}

std::vector<uint32_t> readOnlyLatencyLauncher(size_t arraySizeBytes, size_t strideBytes) { 
    util::hipCheck(hipDeviceReset()); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    readOnlyLatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheLatencyResult measureReadOnlyLatency() {
            auto timings = readOnlyLatencyLauncher(SAMPLE_SIZE * sizeof(uint32_t), sizeof(uint32_t));

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
}