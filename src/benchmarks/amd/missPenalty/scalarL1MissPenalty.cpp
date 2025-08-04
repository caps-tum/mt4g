#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>
#include <cmath>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void scalarL1MissPenaltyKernel(uint32_t *timingResults, size_t steps, uint32_t stride) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE];
    size_t measureLength = util::min(steps, SAMPLE_SIZE);

    uint32_t index = 0;
    // Evict scalar L1 by loading twice the cache size
    for (uint32_t k = 0; k < steps * 2; ++k) {
        #ifdef __HIP_PLATFORM_AMD__
        uint32_t *addr = arr16384AscStride0 + index;
        asm volatile(
            "s_load_dword %0, %1, 0\n\t"
            "s_add_u32 %0, %0, %2\n\t"
            : "+s"(index)
            , "+s"(addr)
            , "+s"(stride)
            :
            : "memory"
        );
        #endif
    }

    index &= 1;
    for (uint32_t k = 0; k < measureLength; ++k) {
        #ifdef __HIP_PLATFORM_AMD__
        uint64_t start, end;
        uint32_t *addr = arr16384AscStride0 + index;
        asm volatile(
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t"

            "s_load_dword %2, %3, 0\n\t"

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %1\n\t"
            
            "s_add_u32 %2, %2, %4\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            : "+s"(start) // uint64_t
            , "+s"(end) // uint64_t
            , "+s"(index) // uint32_t
            , "+s"(addr) // uint32_t*
            , "+s"(stride) // uint32_t
            : : "memory"
        );
        s_timings[k] = end - start;
        #endif
    }

    for (uint32_t k = 0; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = index;
}

std::vector<uint32_t> scalarL1MissPenaltyLauncher(size_t scalarL1CacheSizeBytes, size_t scalarL1FetchGranularityBytes) {
    util::hipCheck(hipDeviceReset());

    size_t steps = scalarL1CacheSizeBytes / scalarL1FetchGranularityBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1MissPenaltyKernel<<<1, 1>>>(d_timingResults, steps, scalarL1FetchGranularityBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin());
    
    util::printVector(timingResultBuffer);

    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        double measureScalarL1MissPenalty(size_t scalarL1CacheSizeBytes, size_t scalarL1FetchGranularityBytes, double scalarL1Latency) {
            auto timings = scalarL1MissPenaltyLauncher(scalarL1CacheSizeBytes, scalarL1FetchGranularityBytes);
            return std::abs(util::average(timings) - scalarL1Latency);
        }
    }
}
