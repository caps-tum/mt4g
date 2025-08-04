#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void textureSharedReadOnlyKernel(hipTextureObject_t tex, const uint32_t* __restrict__ pChaseArrayReadOnly, uint32_t *timingResultsTexture, uint32_t *timingResultsReadOnly, size_t stepsTexture, size_t stepsReadOnly) {
    uint32_t start, end;
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthTexture = util::min(stepsTexture, SAMPLE_SIZE);
    size_t measureLengthReadOnly = util::min(stepsReadOnly, SAMPLE_SIZE);

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        s_timings1[k] = 0;
        s_timings2[k] = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < stepsReadOnly; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(tex, index);
            #endif
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < stepsReadOnly; k++) {
            index = __ldg(&pChaseArrayReadOnly[index]);
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
        timingResultsReadOnly[0] += index >> util::min(stepsReadOnly, 32);

        index = 0;
        for (uint32_t k = 0; k < measureLengthReadOnly; k++) {
            start = clock();
            index = __ldg(&pChaseArrayReadOnly[index]);
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
        for (uint32_t k = 0; k < measureLengthReadOnly; k++) {
            timingResultsReadOnly[k] = s_timings2[k];
        }

        timingResultsReadOnly[0] += index >> util::min(stepsReadOnly, 32);
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> textureSharedReadOnlyLauncher(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, size_t readOnlyCacheSizeBytes,size_t readOnlyFetchGranularityBytes) {
    util::hipCheck(hipDeviceReset()); 

    size_t resultBufferLengthTexture = util::min(textureCacheSizeBytes / textureFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 
    size_t resultBufferLengthReadOnly = util::min(readOnlyCacheSizeBytes / readOnlyFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 

    // Initialize device Arrays
    uint32_t *d_pChaseArrayTexture = util::allocateGPUMemory(util::generatePChaseArray(textureCacheSizeBytes, textureFetchGranularityBytes));
    uint32_t *d_pChaseArrayReadOnly = util::allocateGPUMemory(util::generatePChaseArray(readOnlyCacheSizeBytes, readOnlyFetchGranularityBytes));


    uint32_t *d_timingResultsTexture = util::allocateGPUMemory(resultBufferLengthTexture);
    uint32_t *d_timingResultsReadOnly = util::allocateGPUMemory(resultBufferLengthReadOnly);


    hipTextureObject_t tex = util::createTextureObject(d_pChaseArrayTexture, textureCacheSizeBytes);


    util::hipCheck(hipDeviceSynchronize());
    textureSharedReadOnlyKernel<<<1, 2>>>(tex, d_pChaseArrayReadOnly, d_timingResultsTexture, d_timingResultsReadOnly, textureCacheSizeBytes / textureFetchGranularityBytes, readOnlyCacheSizeBytes / readOnlyFetchGranularityBytes);


    std::vector<uint32_t> timingResultBufferTexture = util::copyFromDevice(d_timingResultsTexture, resultBufferLengthTexture);

    std::vector<uint32_t> timingResultBufferReadOnly = util::copyFromDevice(d_timingResultsReadOnly, resultBufferLengthReadOnly);

    util::hipCheck(hipDeviceReset()); 

    return { timingResultBufferTexture, timingResultBufferReadOnly };
}


namespace benchmark {
    namespace nvidia {
        bool measureTextureAndReadOnlyShared(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, double textureLatency, double textureMissPenalty, size_t readOnlyCacheSizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyLatency, double readOnlyMissPenalty) {
            auto [a, b] = textureSharedReadOnlyLauncher(textureCacheSizeBytes, textureFetchGranularityBytes, readOnlyCacheSizeBytes, readOnlyFetchGranularityBytes);

            double distance1 = std::abs(util::average(a) - textureLatency);
            double distance2 = std::abs(util::average(b) - readOnlyLatency);

            return distance1 - distance2 < textureMissPenalty / 3.0 ||
                   distance1 - distance2 < readOnlyMissPenalty / 3.0;
        }
    }
}