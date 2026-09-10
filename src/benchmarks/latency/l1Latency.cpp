#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <exception>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;// DEFAULT_SAMPLE_SIZE loads should suffice to rule out random flukes

__global__ void l1LatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Warm-up to populate L1
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        index = __allowL1Read(pChaseArray, index);
    }

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t"
        :
        : "l"(s_timings)
    );
    #endif

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        #ifdef __HIP_PLATFORM_AMD__
        uint32_t *addr = pChaseArray + index;
        uint64_t start, end;
        asm volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t"
            "flat_load_dword %1, %3\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %2\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "+s"(start)
            , "+v"(index)
            , "+s"(end)
            , "+v"(addr)
            :
            : "memory"
        );
        #endif
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint32_t start, end;
        uint32_t *addr = pChaseArray + index;
        asm volatile (
            "mov.u32 %0, %%clock;\n\t"
            "ld.global.ca.u32 %1, [%3];\n\t"
            "st.shared.u32 [smem_ptr64], %1;\n\t"
            "mov.u32 %2, %%clock;\n\t"
            : "=r"(start)
            , "=r"(index)
            , "=r"(end)
            : "l"(addr)
            : "memory"
        );
        #endif
        s_timings[i] = end - start;
    }

    for (uint32_t k = 1; k < SAMPLE_SIZE; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = s_timings[0];
}

std::vector<uint32_t> l1LatencyLauncher(size_t arraySizeBytes, size_t strideBytes) { 
    util::hipDeviceReset(); 

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    
    util::hipCheck(hipDeviceSynchronize());
    l1LatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());
    return timingResultBuffer;
}

namespace benchmark {
    CacheLatencyResult measureL1Latency() {
        auto timings = l1LatencyLauncher(SAMPLE_SIZE * sizeof(uint32_t), sizeof(uint32_t));

        CacheLatencyResult result {
            timings,
            util::average(timings),
            util::percentile(timings, 0.5),
            util::percentile(timings, 0.95),
            util::stdev(timings),
            timings.size(),
            SAMPLE_SIZE,
            CYCLE,
            PCHASE
        };

        return result;
    }
}