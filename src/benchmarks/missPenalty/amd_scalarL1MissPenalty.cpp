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
        index = arr16384AscStride0[index] + stride;
    }

    uint32_t sum = index;
    index = 0;
    for (uint32_t k = 0; k < measureLength; ++k) {
        #ifdef __HIP_PLATFORM_AMD__
        uint64_t start, end;
        // Both stride and index may be VGPR (eviction loop uses vector arithmetic).
        // Perform all v_readfirstlane_b32 calls INSIDE the asm block where "=&s" outputs
        // are guaranteed to land in SGPR — avoids the allocator moving them back to VGPR.
        // GFX9: s_load_dword accepts SGPR base (constant symbol) + SGPR byte-offset.
        uint32_t s_strd_scratch, s_byte_scratch;
        asm volatile(
            "v_readfirstlane_b32 %3, %7\n\t"    // s_strd_scratch = stride (VGPR→SGPR)
            "v_readfirstlane_b32 %4, %6\n\t"    // s_byte_scratch = index (VGPR→SGPR)
            "s_lshl_b32 %4, %4, 2\n\t"          // s_byte_scratch = index * 4 (byte offset)
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t"

            "s_load_dword %2, %5, %4\n\t"       // index = arr[index]

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %1\n\t"

            "s_add_u32 %2, %2, %3\n\t"          // index = index + stride
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"

            : "=s"(start)                         // %0 uint64_t
            , "=s"(end)                           // %1 uint64_t
            , "=s"(index)                         // %2 uint32_t
            , "=&s"(s_strd_scratch)               // %3 uint32_t scratch for stride (SGPR)
            , "=&s"(s_byte_scratch)               // %4 uint32_t scratch for byte offset (SGPR)
            : "s"(arr16384AscStride0)            // %5 uint32_t* SGPR pair (constant symbol)
            , "v"(index)                          // %6 uint32_t VGPR (readfirstlane source)
            , "v"(stride)                         // %7 uint32_t VGPR (readfirstlane source)
            : "memory"
        );
        s_timings[k] = end - start;
        #endif
    }

    for (uint32_t k = 1; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = (index + sum & 0x8) >> 2;
}

std::vector<uint32_t> scalarL1MissPenaltyLauncher(size_t scalarL1CacheSizeBytes, size_t scalarL1FetchGranularityBytes) {
    util::hipDeviceReset();

    size_t steps = scalarL1CacheSizeBytes / scalarL1FetchGranularityBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1MissPenaltyKernel<<<1, 1>>>(d_timingResults, steps, scalarL1FetchGranularityBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin());
    
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
