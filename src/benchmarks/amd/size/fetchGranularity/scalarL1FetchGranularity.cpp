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
        uint32_t *addr = arr16384AscStride0 + index;

        asm volatile(
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %0\n\t" // start = clock();

            "s_load_dword %2, %3, 0\n\t" // index = *addr;

            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"
            "s_memtime %1\n\t" // end = clock();

            "s_add_u32 %2, %2, %4\n\t" // index = index + stride

            // Last syncs
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"

            : "+s"(start) // uint64_t
            , "+s"(end) // uint64_t
            , "+s"(index) // uint32_t
            , "+s"(addr) // uint32_t*
            , "+s"(stride) // uint32_t
            :
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
    util::hipCheck(hipDeviceReset());

    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1FetchGranularityKernel<<<1, 1>>>(d_timingResults, fetchGranularityToTestBytes / sizeof(uint32_t));

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    util::hipCheck(hipDeviceReset());
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
