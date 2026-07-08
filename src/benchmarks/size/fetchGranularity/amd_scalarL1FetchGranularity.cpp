#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"
#include "const/constArray16384.hpp"

#include <hip/hip_runtime.h>
#include <optional>
#include <vector>
#include <map>

static constexpr auto MAX_EXPECTED_LINE_SIZE = 256;// B
static constexpr auto SAMPLE_SIZE = 128;// Tries

__global__ void scalarL1FetchGranularityKernel(uint32_t *timingResults, uint32_t stride) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 

    // for some reason index has to be declared here in order for the whole benchmark to not get optimized away
    [[maybe_unused]]uint32_t index = 0;

    for (uint32_t k = 0; k < SAMPLE_SIZE; ++k) {
        #ifdef __HIP_PLATFORM_AMD__
        uint64_t start, end;
        // stride and index may be in VGPR by the time this asm runs.
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
            "s_memtime %0\n\t"                    // start = clock();

            "s_load_dword %2, %5, %4\n\t"       // index = arr[index]

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %1\n\t"                    // end = clock();

            "s_add_u32 %2, %2, %3\n\t"          // index = index + stride

            // Last syncs
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

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        timingResults[k] = s_timings[k];
    }
}


std::vector<uint32_t> scalarL1FetchGranularityLauncher(size_t fetchGranularityToTestBytes) {
    util::hipDeviceReset();

    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1FetchGranularityKernel<<<1, 1>>>(d_timingResults, fetchGranularityToTestBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    util::hipDeviceReset();
    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        CacheSizeResult measureScalarL1FetchGranularity() {
            std::map<size_t, std::vector<uint32_t>> timings;

            for (size_t currentFetchGranularityBytes = sizeof(uint32_t); currentFetchGranularityBytes <= MAX_EXPECTED_LINE_SIZE; currentFetchGranularityBytes += sizeof(uint32_t)) {
                timings[currentFetchGranularityBytes] = scalarL1FetchGranularityLauncher(currentFetchGranularityBytes);
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
}
