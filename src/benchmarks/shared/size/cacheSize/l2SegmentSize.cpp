#include "utils/util.hpp"
#include "benchmarks/benchmark.hpp"

static constexpr auto MIN_EXPECTED_SIZE = 1 * MiB;// 1 * MiB
static constexpr auto MAX_EXPECTED_SIZE = 1024 * MiB;// 64 * MiB
[[maybe_unused]] static constexpr auto ROUNDS = 12;// Rounds
static constexpr auto MEASURE_SIZE = 2048;// Loads

__global__ void l2SegmentSizeKernel(uint32_t *pChaseArray, uint32_t *timingResults, size_t steps) {
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


std::vector<uint32_t> l2SegmentSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
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
    l2SegmentSizeKernel<<<1, util::getMaxThreadsPerBlock(), 0, util::createStreamForCU(0)>>>(d_pChaseArray, d_timingResultBuffer, steps);

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    
    timingResultBuffer.erase(timingResultBuffer.begin());
    //std::cout << "ARR SIZ " << arraySizeBytes << " : " <<  timingResultBuffer[0] << std::endl;
    return timingResultBuffer;
}

namespace benchmark {
    CacheSizeResult measureL2SegmentSize(size_t l2FullSize, size_t l2FetchGranularityBytes) {
        auto [beginBytes, endBytes] = util::findCacheMissRegion(l2SegmentSizeLauncher, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE, l2FetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE/8);
        
        // Adjust initial search range to multiples of CACHE_SIZE_BENCH_RESOLUTION and expand when possible
        std::tie(beginBytes, endBytes) = util::adjustKiBBoundaries(beginBytes, endBytes, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE);

        std::cout << "[L2 Segment Size] Trying Boundaries: " << beginBytes << " - " << endBytes << std::endl;

        std::map<size_t, std::vector<uint32_t>> timings;// = util::runBenchmarkRange(l2SegmentSizeLauncher, 1 * MiB, 13 * MiB, l2FetchGranularityBytes, STEP_SIZE, "L2 Segment Size");

        bool flukeDetected = false;
        size_t flukeCounter = 0;
        bool boundariesRefreshed = false;
        bool aborted = false;

        do {
            // Heuristic: Cache wont get faster with increasing array size, only slower. Thus, you can detect disturbances by checking if the measured timings decreased (significantly) after spiking
            timings = util::runBenchmarkRange(l2SegmentSizeLauncher, beginBytes, endBytes, l2FetchGranularityBytes, CACHE_SIZE_BENCH_RESOLUTION * 16, "L2 Segment Size");

            flukeDetected = util::hasFlukeOccured(timings); // Cache answer times must not decrease again with increasing size, hopefully false most of the time
            if (flukeDetected) {
                ++flukeCounter;
                if (flukeCounter >= 5) {
                    if (!boundariesRefreshed) {
                        std::tie(beginBytes, endBytes) = util::findCacheMissRegion(l2SegmentSizeLauncher, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE, l2FetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE/8);
                        std::tie(beginBytes, endBytes) = util::adjustKiBBoundaries(beginBytes, endBytes, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE);
                        flukeCounter = 0;
                        boundariesRefreshed = true;
                        timings.clear();
                        std::cout << "Benchmark L2 Segment Size fluked 5 times, recalculating boundaries: " << beginBytes << " - " << endBytes << std::endl;
                        continue;
                    } else {
                        aborted = true;
                        flukeDetected = false;
                        std::cout << "Benchmark L2 Segment Size failed despite new boundaries, aborting." << std::endl;
                        break;
                    }
                }

                timings.clear();

                beginBytes -= beginBytes - (CACHE_SIZE_BENCH_RESOLUTION) > endBytes ? 0 : (CACHE_SIZE_BENCH_RESOLUTION); // Prevent underflow
                endBytes += CACHE_SIZE_BENCH_RESOLUTION;

                std::cout << "Benchmark L2 Segment Size measured nonsense, retrying with Boundaries: " << beginBytes << " - " << endBytes << std::endl;
            }

        } while(flukeDetected);


        auto [changePoint, confidence] = util::detectCacheSizeChangePoint(timings);
        if (aborted) {
            confidence = 0;
        }

        auto [clampedSegmentSize, proximity] = util::tryComputeNearestSegmentSize(l2FullSize, changePoint);

        CacheSizeResult result = {
            timings,
            clampedSegmentSize,
            confidence * proximity,
            PCHASE,
            BYTE,
            false
        };
        
        return result;
    }
}