#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void readOnlySharedL1Kernel(const uint32_t* __restrict__ pChaseArrayReadOnly, uint32_t* pChaseArrayL1, uint32_t *timingResultsReadOnly, uint32_t *timingResultsL1, size_t stepsReadOnly, size_t stepsL1) {
    uint32_t start, end;
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthReadOnly = util::min(stepsReadOnly, SAMPLE_SIZE);
    size_t measureLengthL1 = util::min(stepsL1, SAMPLE_SIZE);

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        s_timings1[k] = 0;
        s_timings2[k] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < stepsReadOnly; k++) {
            index = __ldg(&pChaseArrayReadOnly[index]);
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
        timingResultsReadOnly[0] += index >> util::min(stepsReadOnly, 32);

        index = 0;
        //second round
        for (uint32_t k = 0; k < measureLengthReadOnly; k++) {
            start = clock();
            index = __ldg(&pChaseArrayReadOnly[index]);
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
        for (uint32_t k = 0; k < measureLengthReadOnly; k++) {
            timingResultsReadOnly[k] = s_timings1[k];
        }

        timingResultsReadOnly[0] += index >> util::min(stepsReadOnly, 32);
    }

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < measureLengthL1; k++) {
            timingResultsL1[k] = s_timings2[k];
        }

        timingResultsL1[0] += index >> util::min(stepsL1, 32);
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> readOnlySharedL1Launcher(size_t readOnlyCacheSizeBytes, size_t readOnlyFetchGranularityBytes, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes) {
    util::hipCheck(hipDeviceReset()); 

    size_t resultBufferLengthReadOnly = util::min(readOnlyCacheSizeBytes / readOnlyFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t));
    size_t resultBufferLengthL1 = util::min(l1CacheSizeBytes / l1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t));

    // Initialize device Arrays
    uint32_t *d_pChaseArrayReadOnly = util::allocateGPUMemory(util::generatePChaseArray(readOnlyCacheSizeBytes, readOnlyFetchGranularityBytes));
    uint32_t *d_pChaseArrayL1 = util::allocateGPUMemory(util::generatePChaseArray(l1CacheSizeBytes, l1FetchGranularityBytes));

    uint32_t *d_timingResultsReadOnly = util::allocateGPUMemory(resultBufferLengthReadOnly);
    uint32_t *d_timingResultsL1 = util::allocateGPUMemory(resultBufferLengthL1);


    util::hipCheck(hipDeviceSynchronize());
    readOnlySharedL1Kernel<<<1, 2>>>(d_pChaseArrayReadOnly, d_pChaseArrayL1, d_timingResultsReadOnly, d_timingResultsL1, readOnlyCacheSizeBytes / readOnlyFetchGranularityBytes, l1CacheSizeBytes / l1FetchGranularityBytes);


    std::vector<uint32_t> timingResultBufferReadOnly = util::copyFromDevice(d_timingResultsReadOnly, resultBufferLengthReadOnly);
    std::vector<uint32_t> timingResultBufferL1 = util::copyFromDevice(d_timingResultsL1, resultBufferLengthL1);
    

    util::hipCheck(hipDeviceReset()); 

    return { timingResultBufferReadOnly, timingResultBufferL1 };
}


namespace benchmark {
    namespace nvidia {
        bool measureReadOnlyAndL1Shared(size_t readOnlyCacheSizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyLatency, double readOnlyMissPenalty, size_t l1CacheSizeBytes,size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty) {
            auto [a, b] = readOnlySharedL1Launcher(readOnlyCacheSizeBytes, readOnlyFetchGranularityBytes, l1CacheSizeBytes, l1FetchGranularityBytes);

            double distance1 = std::abs(util::average(a) - readOnlyLatency);
            double distance2 = std::abs(util::average(b) - l1Latency);

            return distance1 - distance2 < readOnlyMissPenalty / 3.0 ||
                   distance1 - distance2 < l1MissPenalty / 3.0;
        }
    }
}