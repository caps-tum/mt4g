#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void textureSharedL1Kernel(hipTextureObject_t tex, uint32_t *pChaseArrayL1, uint32_t *timingResultsTexture, uint32_t *timingResultsL1, size_t stepsTexture, size_t stepsL1) {
    uint32_t start, end;
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthTexture = util::min(stepsTexture, SAMPLE_SIZE);
    size_t measureLengthL1 = util::min(stepsL1, SAMPLE_SIZE);

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        s_timings1[k] = 0;
        s_timings2[k] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < stepsTexture; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(tex, index);
            #endif
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
        timingResultsTexture[0] += index >> util::min(stepsTexture, 32);

        index = 0;
        //second round
        for (uint32_t k = 0; k < measureLengthTexture; k++) {
            start = clock();
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(tex, index);
            #endif
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
        for (uint32_t k = 0; k < measureLengthTexture; k++) {
            timingResultsTexture[k] = s_timings1[k];
        }

        timingResultsTexture[0] += index >> util::min(stepsTexture, 32);
    }

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < measureLengthL1; k++) {
            timingResultsL1[k] = s_timings2[k];
        }

        timingResultsL1[0] += index >> util::min(stepsL1, 32);
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> textureSharedL1Launcher(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, size_t l1CacheSizeBytes,size_t l1FetchGranularityBytes) {
    util::hipCheck(hipDeviceReset()); 

    size_t resultBufferLengthTexture = util::min(textureCacheSizeBytes / textureFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 
    size_t resultBufferLengthL1 = util::min(l1CacheSizeBytes / l1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 

    // Initialize device Arrays
    uint32_t *d_pChaseArrayTexture = util::allocateGPUMemory(util::generatePChaseArray(textureCacheSizeBytes, textureFetchGranularityBytes));
    uint32_t *d_pChaseArrayL1 = util::allocateGPUMemory(util::generatePChaseArray(l1CacheSizeBytes, l1FetchGranularityBytes));


    uint32_t *d_timingResultsTexture = util::allocateGPUMemory(resultBufferLengthTexture);
    uint32_t *d_timingResultsL1 = util::allocateGPUMemory(resultBufferLengthL1);


    hipTextureObject_t tex = util::createTextureObject(d_pChaseArrayTexture, textureCacheSizeBytes);


    util::hipCheck(hipDeviceSynchronize());
    textureSharedL1Kernel<<<1, 2>>>(tex, d_pChaseArrayL1, d_timingResultsTexture, d_timingResultsL1, textureCacheSizeBytes / textureFetchGranularityBytes, l1CacheSizeBytes / l1FetchGranularityBytes);


    std::vector<uint32_t> timingResultBufferTexture = util::copyFromDevice(d_timingResultsTexture, resultBufferLengthTexture);

    std::vector<uint32_t> timingResultBufferL1 = util::copyFromDevice(d_timingResultsL1, resultBufferLengthL1);

    util::hipCheck(hipDeviceReset()); 

    return { timingResultBufferTexture, timingResultBufferL1 };
}


namespace benchmark {
    namespace nvidia {
        bool measureTextureAndL1Shared(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, double textureLatency, double textureMissPenalty, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty) {
            auto [a, b] = textureSharedL1Launcher(textureCacheSizeBytes, textureFetchGranularityBytes, l1CacheSizeBytes, l1FetchGranularityBytes);

            double distance1 = std::abs(util::average(a) - textureLatency);
            double distance2 = std::abs(util::average(b) - l1Latency);

            return distance1 - distance2 < textureMissPenalty / 3.0 ||
                   distance1 - distance2 < l1MissPenalty / 3.0;
        }
    }
}