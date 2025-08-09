#include <cstddef>
#include <vector>
#include <map>
#include <tuple>

#include "const/constArray16384.hpp"
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = DEFAULT_SAMPLE_SIZE;// Loads
static constexpr auto GRACE = DEFAULT_GRACE_FACTOR;// Factor
static constexpr auto TESTING_THREADS = 2;

__global__ void constantL1AmountKernel(uint32_t *timingResultsBaseCore, uint32_t *timingResultsTestCore, size_t steps, size_t stride, uint32_t baseCore, uint32_t testCore) {
    // 4 = Amount of Multiprocessor Partitions / SIMDs
    if (__getWarpId() % 4 == testCore / warpSize && threadIdx.x == testCore % warpSize) {
        testCore = threadIdx.x;
    } else if (__getWarpId() % 4 == baseCore / warpSize && threadIdx.x == baseCore % warpSize) {
        baseCore = threadIdx.x;
    } else return;

    uint32_t start, end;
    uint32_t index = 0;
    
    if (threadIdx.x == testCore) {
        index = CONST_ARRAY_SIZE - steps * stride; // testCore should load other arrays than baseCore. 
    }

    __shared__ uint64_t s_timingResultsBaseCore[MEASURE_SIZE];
    __shared__ uint64_t s_timingResultsTestCore[MEASURE_SIZE];


    size_t measureLength = util::min(steps, MEASURE_SIZE);

    // Let the base Core load the first steps values
    if (threadIdx.x == baseCore) {
        for (uint32_t k = 0; k < steps; k++) {
            index = arr16384AscStride0[index] + stride;
        }

        timingResultsBaseCore[0] = index >> util::min(steps, 32);
        index = CONST_ARRAY_SIZE - steps * stride;
    }

    __localBarrier(TESTING_THREADS);
    // If the threads share the same cache physically this will evict all values loaded before
    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < steps; k++) {
            index = arr16384AscStride0[index] + stride;
        }
        timingResultsTestCore[0] = index >> util::min(steps, 32);
        index = CONST_ARRAY_SIZE - steps * stride;
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == baseCore) {
        //second round
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = arr16384AscStride0[index] + stride;
            end = clock();

            s_timingResultsBaseCore[k] = end - start;
        }
        timingResultsBaseCore[0] += index >> util::min(steps, 32);
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == testCore) {
        for (uint32_t k = 0; k < measureLength; k++) {
            start = clock();
            index = arr16384AscStride0[index] + stride;
            end = clock();

            s_timingResultsTestCore[k] = end - start;
        }
        timingResultsTestCore[0] += index >> util::min(steps, 32);
    }

    __localBarrier(TESTING_THREADS);

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

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> constantL1AmountLauncher(size_t constantL1SizeBytes, size_t constantL1FetchGranularityBytes, uint32_t baseCore, uint32_t testCore) {
    util::hipDeviceReset(); 

    constantL1SizeBytes = util::min(constantL1SizeBytes, CONST_ARRAY_SIZE); // Cap at CONST_ARRAY_SIZE, otherwise the benchmark will access illegal addresses and returnt trash values

    size_t resultBufferLength = util::min(constantL1SizeBytes / constantL1FetchGranularityBytes, MEASURE_SIZE);

    uint32_t *d_timingResultsBaseCore = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCore = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    constantL1AmountKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_timingResultsBaseCore, d_timingResultsTestCore, constantL1SizeBytes / constantL1FetchGranularityBytes, constantL1FetchGranularityBytes / sizeof(uint32_t), baseCore, testCore);

    std::vector<uint32_t> baseCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCore, resultBufferLength);
    std::vector<uint32_t> testCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCore, resultBufferLength);

    baseCoreTimingResultsBuffer.erase(baseCoreTimingResultsBuffer.begin());
    testCoreTimingResultsBuffer.erase(testCoreTimingResultsBuffer.begin());

    return { baseCoreTimingResultsBuffer, testCoreTimingResultsBuffer };
}

namespace benchmark {
    namespace nvidia {
        std::optional<uint32_t> measureConstantL1Amount(size_t constantL1SizeBytes, size_t constantL1FetchGranularityBytes, double constantL1MissPenalty) {
            std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCoreToTimingResults;

            if (constantL1SizeBytes > CONST_ARRAY_SIZE) {
                std::cout << "Constant L1 is too large to be benchmarked correctly." << std::endl;
            }

            // Differences are not too great because of CL1.5
            for (uint32_t i = 1; i <= util::getNumberOfCoresPerSM(); i *= 2) {
                auto [baseTimings, testTimings] = testCoreToTimingResults[i] = constantL1AmountLauncher(constantL1SizeBytes, constantL1FetchGranularityBytes, 0, i);
                if (baseTimings[0] == 0 || testTimings[0] == 0) {
                    std::cout << "Error: Base or Test Core timings are zero, indicating an error in the measurement." << std::endl;
                    return std::nullopt;
                }
            }

            return util::getNumberOfCoresPerSM() / util::detectAmountChangePoint(testCoreToTimingResults, constantL1MissPenalty / GRACE);
        }
    }
}