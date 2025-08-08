#include <cstddef>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

static constexpr auto MIN_EXPECTED_SIZE = 256;// Bytes
static constexpr auto MAX_ALLOWED_SIZE = CONST_ARRAY_SIZE;// ~16 * KiB, bandaid fix as larger C1 sizes might falsely detect C1.5 cache borders which is usually above 16 * KiB 

//__attribute__((optimize("O0"), noinline))
__global__ void constantL1SizeKernel(uint32_t *timingResults, size_t steps, uint32_t stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 

    size_t measureLength = util::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t maxIndex = steps * stride;
    uint32_t index = 0;
    uint32_t sum = 0;

    // First round
    for (uint32_t k = 0; k < steps; ++k) {
        index = (arr16384AscStride0[index] + stride) % maxIndex;
        sum += index;
    }

    index &= 1;


    // Second round
    for (uint32_t k = 0; k < measureLength; ++k) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint32_t latency;
        asm volatile(
            ".reg .u32 r_start, r_end, r_tmp;\n\t" 
            ".reg .u64 r_off, r_addr, r_base;\n\t"
            "mov.u64 r_base, arr16384AscStride0;\n\t" 
            "mul.wide.u32 r_off, %1, 4;\n\t"            
            "add.u64 r_addr, r_base, r_off;\n\t"        

            "mov.u32 r_start, %%clock;\n\t"
            "ld.const.u32 r_tmp, [r_addr];\n\t"
            "add.u32 r_tmp, r_tmp, %2;\n\t"
            "rem.u32 r_tmp, r_tmp, %3;\n\t"
            "mov.u32 r_end, %%clock;\n\t"

            "sub.u32 %0, r_end, r_start;\n\t"           
            "mov.u32 %1, r_tmp;\n\t"                    
            : "=r"(latency)
            , "+r"(index)
            : "r"(stride)
            , "r"(maxIndex)
            : "memory"
        );

        s_timings[k] = latency; 
        #endif
    }


    for (uint32_t k = 1; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = sum >> util::min(index, 32);
}

std::vector<uint32_t> constantL1SizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    util::hipDeviceReset();

    constexpr size_t CONST_ARRAY_BYTES = CONST_ARRAY_SIZE * sizeof(uint32_t);
    if (arraySizeBytes > CONST_ARRAY_BYTES) {
        std::cerr << "WARNING: constantL1SizeLauncher array size capped to "
                  << CONST_ARRAY_BYTES << " Bytes" << std::endl;
        arraySizeBytes = CONST_ARRAY_BYTES;
    }
    if (strideBytes == 0) {
        std::cerr << "WARNING: constantL1SizeLauncher stride of 0 is invalid, capping to "
                  << sizeof(uint32_t) << " Bytes" << std::endl;
        strideBytes = sizeof(uint32_t);
    }
    if (strideBytes > CONST_ARRAY_BYTES) {
        std::cerr << "WARNING: constantL1SizeLauncher stride size capped to "
                  << CONST_ARRAY_BYTES << " Bytes" << std::endl;
        strideBytes = CONST_ARRAY_BYTES;
    }
    if (strideBytes > arraySizeBytes) {
        std::cerr << "WARNING: constantL1SizeLauncher stride larger than array size, capping stride" << std::endl;
        strideBytes = arraySizeBytes;
    }

    size_t resultBufferLength = util::min(arraySizeBytes / strideBytes,
                                          MIN_EXPECTED_SIZE / sizeof(uint32_t));
    
    // Allocate GPU VMemory
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    constantL1SizeKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_timingResultBuffer, arraySizeBytes / strideBytes, strideBytes / sizeof(uint32_t));
    
    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);

    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureConstantL1Size(size_t cacheFetchGranularityBytes) {
            auto [beginBytes, endBytes] = util::findCacheMissRegion(constantL1SizeLauncher, MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
        
            std::tie(beginBytes, endBytes) = util::adjustKiBBoundaries(beginBytes, endBytes, cacheFetchGranularityBytes, MAX_ALLOWED_SIZE, true);

            std::cout << "[Constant L1 Size] Trying Boundaries: " << beginBytes << " - " << endBytes << std::endl;

            std::map<size_t, std::vector<uint32_t>> timings;

            bool flukeDetected = false;
            size_t flukeCounter = 0;
            bool boundariesRefreshed = false;
            bool aborted = false;
            do {
                // Heuristic: Cache wont get faster with increasing array size, only slower. Thus, you can detect disturbances by checking if the measured timings decreased (significantly) after spiking
                timings = util::runBenchmarkRange(constantL1SizeLauncher, beginBytes, endBytes, cacheFetchGranularityBytes, cacheFetchGranularityBytes, "Constant L1 Size");

                flukeDetected = util::hasFlukeOccured(timings); // Cache answer times may not decrease again with increasing size, hopefully false most of the time
                if (flukeDetected) {
                    ++flukeCounter;
                    if (flukeCounter >= 5) {
                        if (!boundariesRefreshed) {
                            std::tie(beginBytes, endBytes) = util::findCacheMissRegion(constantL1SizeLauncher, MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
                            std::tie(beginBytes, endBytes) = util::adjustKiBBoundaries(beginBytes, endBytes, cacheFetchGranularityBytes, MAX_ALLOWED_SIZE, true);
                            flukeCounter = 0;
                            boundariesRefreshed = true;
                            timings.clear();
                            std::cout << "Benchmark Constant L1 Size fluked 5 times, recalculating boundaries: " << beginBytes << " - " << endBytes << std::endl;
                            continue;
                        } else {
                            aborted = true;
                            flukeDetected = false;
                            std::cout << "Benchmark Constant L1 Size failed despite new boundaries, aborting." << std::endl;
                            break;
                        }
                    }

                    timings.clear();

                    beginBytes -= beginBytes - cacheFetchGranularityBytes > endBytes ? 0 : cacheFetchGranularityBytes; // Prevent underflow
                    endBytes += cacheFetchGranularityBytes;

                    std::cout << "Benchmark Constant L1 Size measured nonsense, retrying with Boundaries: " << beginBytes << " - " << endBytes << std::endl;
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