#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;
static constexpr auto TESTING_THREADS = 2;

__global__ void textureSharedL1Kernel([[maybe_unused]]hipTextureObject_t tex, uint32_t *pChaseArrayL1, uint32_t *timingResultsTexture, uint32_t *timingResultsL1, size_t stepsTexture, size_t stepsL1) {
    if (blockIdx.x != 0 || threadIdx.x >= 2) return; // Ensure only two threads are used
    uint32_t start, end;
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthTexture = util::min(stepsTexture, SAMPLE_SIZE);
    size_t measureLengthL1 = util::min(stepsL1, SAMPLE_SIZE);

    __localBarrier(TESTING_THREADS);
    
    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < stepsTexture; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(tex, index);
            #endif
        }
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < stepsL1; k++) {
            index = __allowL1Read(pChaseArrayL1, index);
        }
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 0) {
        timingResultsTexture[0] += index;

        //second round
        for (uint32_t k = 0; k < measureLengthTexture; k++) {
            start = clock();
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(tex, index);
            #endif
            end = clock();
            s_timings1[0] += index;
            s_timings1[k] = end - start;
        }
    }

    __localBarrier(TESTING_THREADS);

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t"
        :
        : "l"(s_timings2)
    );
    #endif

    if (threadIdx.x == 1) {
        timingResultsL1[0] += index;

        for (uint32_t k = 0; k < measureLengthL1; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
                uint32_t start, end;
                uint32_t *addr = pChaseArrayL1 + index;
                asm volatile (
                    "mov.u32 %0, %%clock;\n\t"
                    "ld.global.ca.u32 %1, [%3];\n\t"
                    "st.shared.u32 [smem_ptr64], %1;\n\t"
                    "mov.u32 %2, %%clock;\n\t"
                    : "=r"(start)
                    , "=r"(index)
                    , "=r"(end)
                    : "l"(addr)
                    : "memory"
                );
            #endif
            s_timings2[k] = end - start;
        }
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < measureLengthTexture; k++) {
            timingResultsTexture[k] = s_timings1[k];
        }
    }

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < measureLengthL1; k++) {
            timingResultsL1[k] = s_timings2[k];
        }
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> textureSharedL1Launcher(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, size_t l1CacheSizeBytes,size_t l1FetchGranularityBytes) {
    util::hipDeviceReset(); 

    size_t resultBufferLengthTexture = util::min(textureCacheSizeBytes / textureFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 
    size_t resultBufferLengthL1 = util::min(l1CacheSizeBytes / l1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 

    // Initialize device Arrays
    uint32_t *d_pChaseArrayTexture = util::allocateGPUMemory(util::generatePChaseArray(textureCacheSizeBytes, textureFetchGranularityBytes));
    uint32_t *d_pChaseArrayL1 = util::allocateGPUMemory(util::generatePChaseArray(l1CacheSizeBytes, l1FetchGranularityBytes));


    uint32_t *d_timingResultsTexture = util::allocateGPUMemory(resultBufferLengthTexture);
    uint32_t *d_timingResultsL1 = util::allocateGPUMemory(resultBufferLengthL1);


    hipTextureObject_t tex = util::createTextureObject(d_pChaseArrayTexture, textureCacheSizeBytes);


    util::hipCheck(hipDeviceSynchronize());
    textureSharedL1Kernel<<<1, util::getMaxThreadsPerBlock()>>>(tex, d_pChaseArrayL1, d_timingResultsTexture, d_timingResultsL1, textureCacheSizeBytes / textureFetchGranularityBytes, l1CacheSizeBytes / l1FetchGranularityBytes);


    std::vector<uint32_t> timingResultBufferTexture = util::copyFromDevice(d_timingResultsTexture, resultBufferLengthTexture);
    std::vector<uint32_t> timingResultBufferL1 = util::copyFromDevice(d_timingResultsL1, resultBufferLengthL1);

    timingResultBufferTexture.erase(timingResultBufferTexture.begin());
    timingResultBufferL1.erase(timingResultBufferL1.begin());

    return { timingResultBufferTexture, timingResultBufferL1 };
}


namespace benchmark {
    namespace nvidia {
        bool measureTextureAndL1Shared(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, double textureLatency, double textureMissPenalty, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty) {
            auto [timingsTexture, timingsL1] = textureSharedL1Launcher(textureCacheSizeBytes, textureFetchGranularityBytes, l1CacheSizeBytes, l1FetchGranularityBytes);
            
            //std::cout << "Texture Latency: " << util::average(timingsTexture) << ", L1 Latency: " << util::average(timingsL1) << std::endl;
            //std::cout << textureLatency << " " << textureMissPenalty << " " << l1Latency << " " << l1MissPenalty << std::endl;
            
            return util::average(timingsTexture) - textureLatency > textureMissPenalty / SHARED_THRESHOLD || util::average(timingsL1) - l1Latency > l1MissPenalty / SHARED_THRESHOLD;
        }
    }
}