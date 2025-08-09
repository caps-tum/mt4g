#include <cstddef>
#include <vector>
#include <map>
#include <tuple>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = DEFAULT_SAMPLE_SIZE;// Loads
static constexpr auto GRACE = DEFAULT_GRACE_FACTOR;// Factor
static constexpr auto TESTING_THREADS = 2;

__global__ void textureAmountKernel([[maybe_unused]]hipTextureObject_t texBase, [[maybe_unused]]hipTextureObject_t texTest, uint32_t *timingResultsBaseCore, uint32_t *timingResultsTestCore, size_t steps, uint32_t baseCore, uint32_t testCore) {
    // 4 = Amount of Multiprocessor Partitions / SIMDs
    if (__getWarpId() % 4 == testCore / warpSize && threadIdx.x == testCore % warpSize) {
        testCore = threadIdx.x;
    } else if (__getWarpId() % 4 == baseCore / warpSize && threadIdx.x == baseCore % warpSize) {
        baseCore = threadIdx.x;
    } else return;

    __shared__ uint64_t s_timingResultsBaseCore[MEASURE_SIZE];
    __shared__ uint64_t s_timingResultsTestCore[MEASURE_SIZE];

    uint32_t start, end;
    uint32_t index = 0;
    uint32_t measureLength = util::min(steps, MEASURE_SIZE);


    // Let the base Core load the first steps values
    if (threadIdx.x == baseCore) {
        for (uint32_t k = 0; k < steps; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(texBase, index);
            #endif
        }

        s_timingResultsBaseCore[0] += index >> util::min(steps, 32);
        index = 0;
    }

    __localBarrier(TESTING_THREADS);
    // If the threads share the same cache physically this will evict all values loaded before
    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < steps; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(texTest, index);
            #endif
        }

        s_timingResultsTestCore[0] += index >> util::min(steps, 32);
        index = 0;
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == baseCore) {
        //second round
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(texBase, index);
            #endif
            end = clock();

            s_timingResultsBaseCore[k] = end - start;
        }
        
        s_timingResultsBaseCore[0] += index >> util::min(steps, 32);
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(texTest, index);
            #endif
            end = clock();

            s_timingResultsTestCore[k] = end - start;
        }

        s_timingResultsTestCore[0] += index >> util::min(steps, 32);
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == baseCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            timingResultsBaseCore[k] = s_timingResultsBaseCore[k];
        }
    }

    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            timingResultsTestCore[k] = s_timingResultsTestCore[k];
        }
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> textureAmountLauncher(size_t textureSizeBytes, size_t textureFetchGranularityBytes, uint32_t baseCore, uint32_t testCore) {
    util::hipDeviceReset(); 

    std::vector<uint32_t> initializerArray = util::generatePChaseArray(textureSizeBytes, textureFetchGranularityBytes);

    uint32_t *d_pChaseArrayBaseCore = util::allocateGPUMemory(initializerArray);
    uint32_t *d_pChaseArrayTestCore = util::allocateGPUMemory(initializerArray);

    size_t resultBufferLength = util::min(textureSizeBytes / textureFetchGranularityBytes, MEASURE_SIZE);

    uint32_t *d_timingResultsBaseCore = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCore = util::allocateGPUMemory(resultBufferLength);

    hipTextureObject_t texBase = util::createTextureObject(d_pChaseArrayBaseCore, textureSizeBytes);
    hipTextureObject_t texTest = util::createTextureObject(d_pChaseArrayTestCore, textureSizeBytes);

    util::hipCheck(hipDeviceSynchronize());
    textureAmountKernel<<<1, util::getMaxThreadsPerBlock()>>>(texBase, texTest, d_timingResultsBaseCore, d_timingResultsTestCore, textureSizeBytes / textureFetchGranularityBytes, baseCore, testCore);

    std::vector<uint32_t> baseCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCore, resultBufferLength);
    std::vector<uint32_t> testCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCore, resultBufferLength);
    
    baseCoreTimingResultsBuffer.erase(baseCoreTimingResultsBuffer.begin());
    testCoreTimingResultsBuffer.erase(testCoreTimingResultsBuffer.begin());

    return { baseCoreTimingResultsBuffer, testCoreTimingResultsBuffer };
}

// Genaue Recheneinheit ist nicht ermittelbar (scheinbar), eventuell Überlegung starten, wie man das doch ermitteln könnte. Oder sind Threads an Ausführungseinheiten gebunden? Falls nicht ist der aktuelle Ansatz sowieso recht witzlos irgendwie

namespace benchmark {
    namespace nvidia {
        std::optional<uint32_t> measureTextureAmount(size_t textureSizeBytes, size_t textureFetchGranularityBytes, double textureMissPenalty) {
            std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCoreToTimingResults;

            for (uint32_t i = 1; i <= util::getNumberOfCoresPerSM(); i *= 2) {
                auto [baseTimings, testTimings] = testCoreToTimingResults[i] = textureAmountLauncher(textureSizeBytes, textureFetchGranularityBytes, 0, i);
                if (baseTimings[0] == 0 || testTimings[0] == 0) {
                    std::cout << "Error: Base or Test Core timings are zero, indicating an error in the measurement." << std::endl;
                    return std::nullopt;
                }
            }
            
            return util::getNumberOfCoresPerSM() / util::detectAmountChangePoint(testCoreToTimingResults, textureMissPenalty / GRACE);
        }
    }
}