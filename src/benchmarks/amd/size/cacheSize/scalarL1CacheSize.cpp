#include <cstddef>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

static constexpr auto MIN_EXPECTED_SIZE = 1024;// Bytes
static constexpr auto MAX_ALLOWED_SIZE = CONST_ARRAY_SIZE * sizeof(uint32_t);// ~64 * KiB

//__attribute__((optimize("O0"), noinline))
__global__ void scalarL1SizeKernel(uint32_t *timingResults, size_t steps, uint32_t stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 

    size_t measureLength = util::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t index = 0;

    // First round
    for (uint32_t k = 0; k < steps; ++k) {
        index = arr16384AscStride0[index] + stride;
    }

    uint32_t sum = index;
    index = 0;

    // Second round
    for (uint32_t k = 0; k < measureLength; ++k) {
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



    for (uint32_t k = 1; k < measureLength; ++k) { // from 1 because idx 0 is trash, to be discussed
        timingResults[k] = s_timings[k];
    }

    //timingResults[0] =  (end - start) / measureLength;
    timingResults[0] = (index + sum & 0x8) >> 2;
    //timingResults[2] =  sum;
}

std::vector<uint32_t> scalarL1SizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    util::hipDeviceReset();

    //std::cout << arraySizeBytes << std::endl;

    size_t resultBufferLength = util::min(arraySizeBytes / strideBytes, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1SizeKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_timingResultBuffer, arraySizeBytes / strideBytes, strideBytes / sizeof(uint32_t));

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    timingResultBuffer.erase(timingResultBuffer.begin());
    util::hipDeviceReset();

    return timingResultBuffer;
}

namespace benchmark {
    namespace amd {
        CacheSizeResult measureScalarL1Size(size_t cacheFetchGranularityBytes) {
            auto [beginBytes, endBytes] = util::findCacheMissRegion(scalarL1SizeLauncher, MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
        
            // Adjust initial search range to multiples of CACHE_SIZE_BENCH_RESOLUTION and expand when possible
            std::tie(beginBytes, endBytes) =
                util::adjustKiBBoundaries(beginBytes, endBytes,
                                           MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE,
                                           true);

            std::cout << "[Scalar L1 Size] Trying Boundaries: " << beginBytes << " - " << endBytes << std::endl;

            std::map<size_t, std::vector<uint32_t>> timings;

            bool flukeDetected = false;
            size_t flukeCounter = 0;
            bool boundariesRefreshed = false;
            bool aborted = false;
            do {
                // Heuristic: Cache wont get faster with increasing array size, only slower. Thus, you can detect disturbances by checking if the measured timings decreased (significantly) after spiking
                timings = util::runBenchmarkRange(scalarL1SizeLauncher, beginBytes, endBytes, cacheFetchGranularityBytes, CACHE_SIZE_BENCH_RESOLUTION, "Scalar L1 Size");

                flukeDetected = util::hasFlukeOccured(timings); // Cache answer times may not decrease again with increasing size, hopefully false most of the time
                if (flukeDetected) {
                    ++flukeCounter;
                    if (flukeCounter >= 5) {
                        if (!boundariesRefreshed) {
                            std::tie(beginBytes, endBytes) = util::findCacheMissRegion(scalarL1SizeLauncher, MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
                            std::tie(beginBytes, endBytes) =
                                util::adjustKiBBoundaries(beginBytes, endBytes,
                                                           MIN_EXPECTED_SIZE,
                                                           MAX_ALLOWED_SIZE,
                                                           true);
                            flukeCounter = 0;
                            boundariesRefreshed = true;
                            timings.clear();
                            std::cout << "Benchmark Scalar L1 Size fluked 5 times, recalculating boundaries: " << beginBytes << " - " << endBytes << std::endl;
                            continue;
                        } else {
                            aborted = true;
                            flukeDetected = false;
                            std::cout << "Benchmark Scalar L1 Size failed despite new boundaries, aborting." << std::endl;
                            break;
                        }
                    }

                    timings.clear();

                    beginBytes -= beginBytes - (CACHE_SIZE_BENCH_RESOLUTION) > endBytes ? 0 : (CACHE_SIZE_BENCH_RESOLUTION); // Prevent underflow
                    endBytes += CACHE_SIZE_BENCH_RESOLUTION;

                    std::cout << "Benchmark Scalar L1 Size measured nonsense, retrying with Boundaries: " << beginBytes << " - " << endBytes << std::endl;
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
}