#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

#include <vector>

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void scalarL1LatencyKernel(uint32_t *timingResults) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    uint32_t index = 0;
    uint32_t sum = 0;

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        index = arr16384AscStride0[index] + 1;
        sum += index;
    }

    index &= 1;
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

            "s_add_u32 %2, %2, 1\n\t" // index = index + 1

            // Last syncs
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_waitcnt vmcnt(0)\n\t"

            : "+s"(start) // uint64_t
            , "+s"(end) // uint64_t
            , "+s"(index) //uint32_t
            , "+s"(addr) // uint32_t*
            :
            : "memory"
        );

        s_timings[k] = end - start;
        #endif
    }

    for (uint32_t k = 0; k < SAMPLE_SIZE; ++k) {
        timingResults[k] = s_timings[k];
    }

    //timingResults[0] =  (end - start) / measureLength;
    timingResults[0] += (index + sum & 0x8) >> 2;
    //timingResults[2] =  sum;
}

std::vector<uint32_t> scalarL1LatencyLauncher() {
    util::hipCheck(hipDeviceReset());

    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1LatencyKernel<<<1, 1>>>(d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    timingResultBuffer.erase(timingResultBuffer.begin());
    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        CacheLatencyResult measureScalarL1Latency() {
            auto timings = scalarL1LatencyLauncher();

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

