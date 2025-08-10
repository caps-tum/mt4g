#include <cstddef>
#include <vector>
#include <map>
#include <tuple>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = DEFAULT_SAMPLE_SIZE;// Loads
static constexpr auto GRACE = DEFAULT_GRACE_FACTOR;// Factor
static constexpr auto TESTING_THREADS = 2;

__global__ void readOnlyAmountKernel(const uint32_t* __restrict__ pChaseArrayBase, const uint32_t* __restrict__ pChaseArrayTest, uint32_t *timingResultsBaseCore, uint32_t *timingResultsTestCore, size_t steps, uint32_t baseCore, uint32_t testCore) {
    if (__getWarpId() == testCore / warpSize && threadIdx.x % warpSize == testCore % warpSize) {
        // printf("testCore: %d __getWarpId: %d, threadIdx.x: %d\n", testCore, __getWarpId(), threadIdx.x);
        testCore = threadIdx.x;
    } else if (__getWarpId() == baseCore / warpSize && threadIdx.x % warpSize == baseCore % warpSize) {
        // printf("baseCore: %d __getWarpId: %d, threadIdx.x: %d\n", baseCore, __getWarpId(), threadIdx.x);
        baseCore = threadIdx.x;
    } else return;

    __shared__ uint64_t s_timingResultsBaseCore[MEASURE_SIZE];
    __shared__ uint64_t s_timingResultsTestCore[MEASURE_SIZE];

    uint32_t core = __getPhysicalCUId();
    uint32_t warp = __getWarpId();  

    uint32_t start, end;
    uint32_t index = 0;
    uint32_t measureLength = util::min(steps, MEASURE_SIZE);

    
    // Let the base Core load the first steps values
    if (threadIdx.x == baseCore) {
        for (uint32_t k = 0; k < steps; k++) {
            index = __ldg(&pChaseArrayBase[index]);
        }

        s_timingResultsBaseCore[0] += index >> util::min(steps, 32);
        index = 0;
    }

    __localBarrier(TESTING_THREADS);
    // If the threads share the same cache physically 
    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < steps; k++) {
            index = __ldg(&pChaseArrayTest[index]);
        }

        timingResultsTestCore[0] = index;
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == baseCore) {
        //second round
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = __ldg(&pChaseArrayBase[index]);
            end = clock();

            s_timingResultsBaseCore[k] = end - start;
        }
        
        timingResultsBaseCore[0] += index;
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = __ldg(&pChaseArrayTest[index]);
            end = clock();

            s_timingResultsTestCore[k] = end - start;
        }

        timingResultsTestCore[0] += index;
    }

    __localBarrier(TESTING_THREADS);

    if (core != __getPhysicalCUId() || warp != __getWarpId()) {
        return; // Not on the same SM anymore
    }

    if (threadIdx.x == baseCore) {
        for (uint32_t k = 1; k < measureLength; k++) {
            timingResultsBaseCore[k] = s_timingResultsBaseCore[k];
        }
    }

    if (threadIdx.x == testCore) {
        for (uint32_t k = 1; k < measureLength; k++) {
            timingResultsTestCore[k] = s_timingResultsTestCore[k];
        }
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> readOnlyAmountLauncher(size_t readOnlySizeBytes, size_t readOnlyFetchGranularityBytes, uint32_t baseCore, uint32_t testCore) {
    util::hipDeviceReset(); 

    std::vector<uint32_t> initializerArray = util::generatePChaseArray(readOnlySizeBytes, readOnlyFetchGranularityBytes);

    uint32_t *d_pChaseArrayBaseCore = util::allocateGPUMemory(initializerArray);
    uint32_t *d_pChaseArrayTestCore = util::allocateGPUMemory(initializerArray);

    size_t resultBufferLength = util::min(readOnlySizeBytes / readOnlyFetchGranularityBytes, MEASURE_SIZE);

    uint32_t *d_timingResultsBaseCore = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCore = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    readOnlyAmountKernel<<<1, util::getNumberOfCoresPerSM()>>>(d_pChaseArrayBaseCore, d_pChaseArrayTestCore, d_timingResultsBaseCore, d_timingResultsTestCore, readOnlySizeBytes / readOnlyFetchGranularityBytes, baseCore, testCore);

    std::vector<uint32_t> baseCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCore, resultBufferLength);
    std::vector<uint32_t> testCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCore, resultBufferLength);
    
    baseCoreTimingResultsBuffer.erase(baseCoreTimingResultsBuffer.begin());
    testCoreTimingResultsBuffer.erase(testCoreTimingResultsBuffer.begin());

    return { baseCoreTimingResultsBuffer, testCoreTimingResultsBuffer };
}

namespace benchmark {
    namespace nvidia {
        std::optional<uint32_t> measureReadOnlyAmount(size_t readOnlySizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyMissPenalty) {
            std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCoreToTimingResults;

            for (uint32_t i = 1; i <= util::getNumberOfCoresPerSM(); i *= 2) { 
                auto [baseTimings, testTimings] = testCoreToTimingResults[i] = readOnlyAmountLauncher(readOnlySizeBytes, readOnlyFetchGranularityBytes, 0, i);
                if ( i<util::getNumberOfCoresPerSM() && (baseTimings[0] == 0 || testTimings[0] == 0)) {
                    std::cout << "Error: Base or Test Core timings are zero, indicating an error in the measurement." << std::endl;
                    return std::nullopt;
                }
            }

            return util::getNumberOfCoresPerSM() / util::detectAmountChangePoint(testCoreToTimingResults, readOnlyMissPenalty / GRACE);
        }
    }
}