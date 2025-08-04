#include <cstddef>
#include <hip/hip_runtime.h>

#include "const/constArray16384.hpp"
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void constantL1SharedL1Kernel(uint32_t* pChaseArrayL1, uint32_t *timingResultsConstantL1, uint32_t *timingResultsL1, size_t stepsConstantL1, size_t stepsL1, size_t constantL1stride) {
    uint32_t start, end;
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthConstantL1 = util::min(stepsConstantL1, SAMPLE_SIZE);
    size_t measureLengthL1 = util::min(stepsL1, SAMPLE_SIZE);

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        s_timings1[k] = 0;
        s_timings2[k] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < stepsConstantL1; k++) {
            index = arr16384AscStride0[index] + constantL1stride;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < stepsL1; k++) {
            index = __allowL1Read(pChaseArrayL1, index);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        timingResultsConstantL1[0] += index >> util::min(stepsConstantL1, 32);

        index = 0;
        //second round
        for (uint32_t k = 0; k < measureLengthConstantL1; k++) {
            start = clock();
            index = arr16384AscStride0[index] + constantL1stride;
            end = clock();
            s_timings1[k] = end - start;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        timingResultsL1[0] += index >> util::min(stepsL1, 32);

        index = 0;
        for (uint32_t k = 0; k < measureLengthL1; k++) {
            start = clock();
            index = __allowL1Read(pChaseArrayL1, index);
            end = clock();
            s_timings2[k] = end - start;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < measureLengthConstantL1; k++) {
            timingResultsConstantL1[k] = s_timings1[k];
        }

        timingResultsConstantL1[0] += index >> util::min(stepsConstantL1, 32);
    }

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < measureLengthL1; k++) {
            timingResultsL1[k] = s_timings2[k];
        }

        timingResultsL1[0] += index >> util::min(stepsL1, 32);
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> constantL1SharedL1Launcher(size_t constantL1CacheSizeBytes, size_t constantL1FetchGranularityBytes, size_t l1CacheSizeBytes,size_t l1FetchGranularityBytes) {
    util::hipCheck(hipDeviceReset()); 

    size_t resultBufferLengthConstantL1 = util::min(constantL1CacheSizeBytes / constantL1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 
    size_t resultBufferLengthL1 = util::min(l1CacheSizeBytes / l1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 

    // Initialize device Arrays
    uint32_t *d_pChaseArrayL1 = util::allocateGPUMemory(util::generatePChaseArray(l1CacheSizeBytes, l1FetchGranularityBytes));

    uint32_t *d_timingResultsConstantL1 = util::allocateGPUMemory(resultBufferLengthConstantL1);
    uint32_t *d_timingResultsL1 = util::allocateGPUMemory(resultBufferLengthL1);


    util::hipCheck(hipDeviceSynchronize());
    constantL1SharedL1Kernel<<<1, 2>>>(d_pChaseArrayL1, d_timingResultsConstantL1, d_timingResultsL1, constantL1CacheSizeBytes / constantL1FetchGranularityBytes, l1CacheSizeBytes / l1FetchGranularityBytes, constantL1FetchGranularityBytes / sizeof(uint32_t));


    std::vector<uint32_t> timingResultBufferConstantL1 = util::copyFromDevice(d_timingResultsConstantL1, resultBufferLengthConstantL1);
    std::vector<uint32_t> timingResultBufferL1 = util::copyFromDevice(d_timingResultsL1, resultBufferLengthL1);


    util::hipCheck(hipDeviceReset()); 

    return { timingResultBufferConstantL1, timingResultBufferL1 };
}


namespace benchmark {
    namespace nvidia {
        bool measureConstantL1AndL1Shared(size_t constantL1CacheSizeBytes, size_t constantL1FetchGranularityBytes, double constantL1Latency, double constantL1MissPenalty, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty) {
            auto [a, b] = constantL1SharedL1Launcher(constantL1CacheSizeBytes, constantL1FetchGranularityBytes, l1CacheSizeBytes, l1FetchGranularityBytes);

            double distance1 = std::abs(util::average(a) - constantL1Latency);
            double distance2 = std::abs(util::average(b) - l1Latency);

            return distance1 - distance2 < constantL1MissPenalty / 3.0 ||
                   distance1 - distance2 < l1MissPenalty / 3.0;
        }
    }
}