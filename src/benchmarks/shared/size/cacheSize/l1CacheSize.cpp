#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <cmath>

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_EXPECTED_SIZE = 1048576;// 1024 Bytes * 1024 Bytes = 1 * MiB

//__attribute__((optimize("O0"), noinline))
__global__ void l1SizeKernel(uint32_t *pChaseArray, uint32_t *timingResults, size_t steps) {
    // s_timings[0] is undefined, as we use it to prevent compiler optimizations / latency hiding
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    
    uint32_t index = 0;
    size_t measureLength = util::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    // First Round to (hopefully) fill L1 Cache, GLC=0 / .ca hint
    for (uint32_t i = 0; i < steps; ++i) {
        index = __allowL1Read(pChaseArray, index);
    }

    // Second Round to (hopefully) load Data from vL1d, GLC=0
    #ifdef __HIP_PLATFORM_NVIDIA__ // Prepare &s_timings[0] to be used in PTX to avoid latency hiding. "smem_ptr64" will contain PTX friendly address 
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t" 
        :
        : "l"(s_timings) // __shared__ uint32_t*
    );
    #endif
    for (uint32_t i = 0; i < measureLength; ++i) {
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

    for (uint32_t k = 1; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] += s_timings[0] >> util::min(steps, 32);
}

std::vector<uint32_t> l1SizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    util::hipCheck(hipDeviceReset());

    size_t stridedLength = arraySizeBytes / strideBytes;
    size_t resultBufferLength = std::min(stridedLength, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    

    util::hipCheck(hipDeviceSynchronize());
    l1SizeKernel<<<1, 1>>>(d_pChaseArray, d_timingResultBuffer, stridedLength);

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    
    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    CacheSizeResult measureL1Size(size_t cacheFetchGranularityBytes) { // vL1d for AMD, to be precise
        auto [beginBytes, endBytes] = util::findCacheMissRegion(l1SizeLauncher, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);

        // Adjust initial search range to multiples of CACHE_SIZE_BENCH_RESOLUTION and expand when possible
        std::tie(beginBytes, endBytes) = util::adjustKiBBoundaries(beginBytes, endBytes, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE);

        std::cout << "[L1 Size] Trying Boundaries: " << beginBytes << " - " << endBytes << std::endl;

        std::map<size_t, std::vector<uint32_t>> timings;

        bool flukeDetected = false;
        size_t flukeCounter = 0;
        bool boundariesRefreshed = false;
        bool aborted = false;
        do {
            // Heuristic: Cache wont get faster with increasing array size, only slower. Thus, you can detect disturbances by checking if the measured timings decreased (significantly) after spiking
            timings = util::runBenchmarkRange(l1SizeLauncher, beginBytes, endBytes, cacheFetchGranularityBytes, CACHE_SIZE_BENCH_RESOLUTION, "L1 Size");

            flukeDetected = util::hasFlukeOccured(timings); // Cache answer times may not decrease again with increasing size, hopefully false most of the time
            if (flukeDetected) {
                ++flukeCounter;
                if (flukeCounter >= 5) {
                    if (!boundariesRefreshed) {
                        std::tie(beginBytes, endBytes) = util::findCacheMissRegion(l1SizeLauncher, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
                        std::tie(beginBytes, endBytes) = util::adjustKiBBoundaries(beginBytes, endBytes, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE);
                        flukeCounter = 0;
                        boundariesRefreshed = true;
                        timings.clear();
                        std::cout << "Benchmark L1 Size fluked 5 times, recalculating boundaries: " << beginBytes << " - " << endBytes << std::endl;
                        continue;
                    } else {
                        aborted = true;
                        flukeDetected = false;
                        std::cout << "Benchmark L1 Size failed despite new boundaries, aborting." << std::endl;
                        break;
                    }
                }

                timings.clear();

                beginBytes -= beginBytes - (CACHE_SIZE_BENCH_RESOLUTION) > endBytes ? 0 : (CACHE_SIZE_BENCH_RESOLUTION); // Prevent underflow
                endBytes += CACHE_SIZE_BENCH_RESOLUTION;

                std::cout << "Benchmark L1 Size measured nonsense, retrying with Boundaries: " << beginBytes << " - " << endBytes << std::endl;
            }
        } while(flukeDetected);


        auto [changePoint, confidence] = util::detectCacheSizeChangePoint(timings);
        if (aborted) {
            confidence = 0;
        }

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