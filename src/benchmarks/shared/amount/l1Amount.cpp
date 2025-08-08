#include <cstddef>
#include <vector>
#include <map>
#include <cstdint>
#include <tuple>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = DEFAULT_SAMPLE_SIZE;// Loads
static constexpr auto GRACE = DEFAULT_GRACE_FACTOR;// Factor
static constexpr auto TESTING_THREADS = 2;

__global__ void l1AmountKernel(uint32_t* pChaseArrayBaseCore, uint32_t *pChaseArrayTestCore, uint32_t *timingResultsBaseCore, uint32_t *timingResultsTestCore, size_t steps, uint32_t baseCore, uint32_t testCore) {
    // 4 = Amount of Multiprocessor Partitions / SIMDs
    if (__getWarpId() % 4 == testCore / 32 && threadIdx.x == testCore % 32) {
        testCore = threadIdx.x;
    } else if (__getWarpId() % 4 == baseCore / 32 && threadIdx.x == baseCore % 32) {
        baseCore = threadIdx.x;
    } 
    
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
            index = __allowL1Read(pChaseArrayBaseCore, index);
        }

        timingResultsBaseCore[0] = index;
    }

    __localBarrier(TESTING_THREADS);
    // If the threads share the same cache physically this will evict all values loaded before
    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < steps; k++) {
            index = __allowL1Read(pChaseArrayTestCore, index);
        }

        timingResultsTestCore[0] = index;
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == baseCore) {
        //second round
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = __allowL1Read(pChaseArrayBaseCore, index);
            end = clock();

            s_timingResultsBaseCore[k] = end - start;
        }
        
        timingResultsBaseCore[0] += index;
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = __allowL1Read(pChaseArrayTestCore, index);
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

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> l1AmountLauncher(size_t l1SizeBytes, size_t l1FetchGranularityBytes, uint32_t baseCore, uint32_t testCore) {
    util::hipDeviceReset(); 

    std::vector<uint32_t> initializerArray = util::generatePChaseArray(l1SizeBytes, l1FetchGranularityBytes);

    uint32_t *d_pChaseArrayBaseCore = util::allocateGPUMemory(initializerArray);
    uint32_t *d_pChaseArrayTestCore = util::allocateGPUMemory(initializerArray);

    size_t resultBufferLength = util::min(l1SizeBytes / l1FetchGranularityBytes, MEASURE_SIZE);

    uint32_t *d_timingResultsBaseCore = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCore = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    l1AmountKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_pChaseArrayBaseCore, d_pChaseArrayTestCore, d_timingResultsBaseCore, d_timingResultsTestCore, l1SizeBytes / l1FetchGranularityBytes, baseCore, testCore);

    std::vector<uint32_t> baseCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCore, resultBufferLength);
    std::vector<uint32_t> testCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCore, resultBufferLength);

    baseCoreTimingResultsBuffer.erase(baseCoreTimingResultsBuffer.begin());
    testCoreTimingResultsBuffer.erase(testCoreTimingResultsBuffer.begin());

    return { baseCoreTimingResultsBuffer, testCoreTimingResultsBuffer };
}

namespace benchmark {
    std::optional<uint32_t> measureL1Amount(size_t l1SizeBytes, size_t l1FetchGranularityBytes, double l1MissPenalty) {
        std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCoreToTimingResults;

        for (uint32_t i = 1; i <= util::getNumberOfCoresPerSM(); i *= 2) { 
            auto [baseTimings, testTimings] = testCoreToTimingResults[i] = l1AmountLauncher(l1SizeBytes, l1FetchGranularityBytes, 0, i);
            if (baseTimings[0] == 0 || testTimings[0] == 0) {
                std::cout << "Error: Base or Test Core timings are zero, indicating an error in the measurement." << std::endl;
                return std::nullopt;
            }
        }

        return util::getNumberOfCoresPerSM() / util::detectAmountChangePoint(testCoreToTimingResults, l1MissPenalty / GRACE);
    }
}