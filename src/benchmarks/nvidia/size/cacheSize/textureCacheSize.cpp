#include <cstddef>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MIN_EXPECTED_SIZE = 1024;// B
static constexpr auto MAX_EXPECTED_SIZE = 128 * 1024;// 128 * KiB

__global__ void textureSizeKernel(hipTextureObject_t tex, uint32_t *timingResults, size_t length) {
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[MIN_EXPECTED_SIZE / sizeof(uint32_t)];

    size_t measureLength = util::min(length, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t start, end;
    uint32_t index = 0;

    for (uint32_t  k = 0; k < measureLength; k++) {
        s_index[k] = 0;
        s_timings[k] = 0;
    }

    // First round
    for (uint32_t k = 0; k < length; k++) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        index = tex1Dfetch<uint32_t>(tex, index);
        #endif
    }

    // Second round
    for (uint32_t k = 0; k < measureLength; k++) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        start = clock();
        index = tex1Dfetch<uint32_t>(tex, index);
        s_index[k] = index;
        end = clock();
        s_timings[k] = end - start;
        #endif
    }

    for (uint32_t k = 0; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
        s_index[0] += s_index[k];
    }

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}

std::vector<uint32_t> textureSizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    util::hipCheck(hipDeviceReset());

    size_t arraySize = arraySizeBytes / sizeof(uint32_t);
    size_t stride = strideBytes / sizeof(uint32_t);
    size_t steps = arraySize / stride + (arraySize % stride != 0 ? 1 : 0);
    size_t resultBufferLength = std::min(steps, MIN_EXPECTED_SIZE / sizeof(uint32_t)); 
    
    // Allocate GPU VMemory
    uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(arraySizeBytes, strideBytes));
    
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    
    hipTextureObject_t tex = util::createTextureObject(d_pChaseArray, arraySizeBytes);
    

    util::hipCheck(hipDeviceSynchronize());
    textureSizeKernel<<<1, 1>>>(tex, d_timingResultBuffer, steps);
    util::hipCheck(hipDeviceSynchronize());


    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    

    util::hipCheck(hipDeviceReset());

    return timingResultBuffer;
}

namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureTextureSize(size_t cacheFetchGranularityBytes) {
            auto [beginBytes, endBytes] = util::findCacheMissRegion(textureSizeLauncher, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
            
            // Adjust initial search range to multiples of CACHE_SIZE_BENCH_RESOLUTION and expand when possible
            std::tie(beginBytes, endBytes) =
                util::adjustKiBBoundaries(beginBytes, endBytes,
                                           MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE);

            std::cout << "[Texture Size] Trying Boundaries: " << beginBytes << " - " << endBytes << std::endl;

            std::map<size_t, std::vector<uint32_t>> timings;

            bool flukeDetected = false;
            size_t flukeCounter = 0;
            bool boundariesRefreshed = false;
            bool aborted = false;

            do {
                // Heuristic: Cache wont get faster with increasing array size, only slower. Thus, you can detect disturbances by checking if the measured timings decreased (significantly) after spiking
                timings = util::runBenchmarkRange(textureSizeLauncher, beginBytes, endBytes, cacheFetchGranularityBytes, CACHE_SIZE_BENCH_RESOLUTION, "Texture Size");

                util::pipeMapToPython(timings, std::to_string(flukeCounter) + " Texture");
                flukeDetected = util::hasFlukeOccured(timings); // Cache answer times may not decrease again with increasing size, hopefully false most of the time
                if (flukeDetected) {
                    ++flukeCounter;
                    if (flukeCounter >= 5) {
                        if (!boundariesRefreshed) {
                            std::tie(beginBytes, endBytes) = util::findCacheMissRegion(textureSizeLauncher, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
                            std::tie(beginBytes, endBytes) =
                                util::adjustKiBBoundaries(beginBytes, endBytes, MIN_EXPECTED_SIZE, MAX_EXPECTED_SIZE);
                            flukeCounter = 0;
                            boundariesRefreshed = true;
                            timings.clear();
                            std::cout << "Benchmark Texture Size fluked 5 times, recalculating boundaries: " << beginBytes << " - " << endBytes << std::endl;
                            continue;
                        } else {
                            aborted = true;
                            flukeDetected = false;
                            std::cout << "Benchmark Texture Size failed despite new boundaries, aborting." << std::endl;
                            break;
                        }
                    }

                    timings.clear();

                    beginBytes -= beginBytes - (CACHE_SIZE_BENCH_RESOLUTION) > endBytes ? 0 : (CACHE_SIZE_BENCH_RESOLUTION); // Prevent underflow
                    endBytes += CACHE_SIZE_BENCH_RESOLUTION;

                    std::cout << "Benchmark Texture Size measured nonsense, retrying with Boundaries: " << beginBytes << " - " << endBytes << std::endl;
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