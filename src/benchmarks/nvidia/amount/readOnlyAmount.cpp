#include <cstddef>
#include <vector>
#include <map>
#include <tuple>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = DEFAULT_SAMPLE_SIZE;// Loads
static constexpr auto GRACE = DEFAULT_GRACE_FACTOR;// Factor

__global__ void readOnlyAmountKernel(const uint32_t* __restrict__ pChaseArrayBase, const uint32_t* __restrict__ pChaseArrayTest, uint32_t *timingResultsBaseCore, uint32_t *timingResultsTestCore, size_t steps, uint32_t baseCore, uint32_t testCore) {
    __shared__ uint64_t s_timingResultsBaseCore[MEASURE_SIZE];
    __shared__ uint64_t s_timingResultsTestCore[MEASURE_SIZE];

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

    __syncthreads();
    // If the threads share the same cache physically 
    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < steps; k++) {
            index = __ldg(&pChaseArrayTest[index]);
        }

        s_timingResultsTestCore[0] += index >> util::min(steps, 32);
        index = 0;
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        //second round
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = __ldg(&pChaseArrayBase[index]);
            end = clock();

            s_timingResultsBaseCore[k] = end - start;
        }
        
        s_timingResultsBaseCore[0] += index >> util::min(steps, 32);
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = __ldg(&pChaseArrayTest[index]);
            end = clock();

            s_timingResultsTestCore[k] = end - start;
        }

        s_timingResultsTestCore[0] += index >> util::min(steps, 32);
    }

    __syncthreads();

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

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> readOnlyAmountLauncher(size_t readOnlySizeBytes, size_t readOnlyFetchGranularityBytes, uint32_t baseCore, uint32_t testCore) {
    util::hipCheck(hipDeviceReset()); 

    std::vector<uint32_t> initializerArray = util::generatePChaseArray(readOnlySizeBytes, readOnlyFetchGranularityBytes);

    uint32_t *d_pChaseArrayBaseCore = util::allocateGPUMemory(initializerArray);
    uint32_t *d_pChaseArrayTestCore = util::allocateGPUMemory(initializerArray);

    size_t resultBufferLength = util::min(readOnlySizeBytes / readOnlyFetchGranularityBytes, MEASURE_SIZE);

    uint32_t *d_timingResultsBaseCore = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCore = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    readOnlyAmountKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_pChaseArrayBaseCore, d_pChaseArrayTestCore, d_timingResultsBaseCore, d_timingResultsTestCore, readOnlySizeBytes / readOnlyFetchGranularityBytes, baseCore, testCore);

    std::vector<uint32_t> baseCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCore, resultBufferLength);
    std::vector<uint32_t> testCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCore, resultBufferLength);

    return { baseCoreTimingResultsBuffer, testCoreTimingResultsBuffer };
}

namespace benchmark {
    namespace nvidia {
        uint32_t measureReadOnlyAmount(size_t readOnlySizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyMissPenalty) {
            std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCoreToTimingResults;


            for (uint32_t i = 1; i <= util::getNumberOfCoresPerSM(); i *= 2) { 
                testCoreToTimingResults[i] = readOnlyAmountLauncher(readOnlySizeBytes, readOnlyFetchGranularityBytes, 0, i);
            }

            return util::getNumberOfCoresPerSM() / util::detectAmountChangePoint(testCoreToTimingResults, readOnlyMissPenalty / GRACE);
        }
    }
}