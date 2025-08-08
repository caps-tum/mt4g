#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;
static constexpr auto TESTING_THREADS = 2;

__global__ void textureSharedReadOnlyKernel([[maybe_unused]]hipTextureObject_t tex, const uint32_t* __restrict__ pChaseArrayReadOnly, uint32_t *timingResultsTexture, uint32_t *timingResultsReadOnly, size_t stepsTexture, size_t stepsReadOnly) {
    if (blockIdx.x != 0 || threadIdx.x >= 2) return; // Ensure only two threads are used
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthTexture = util::min(stepsTexture, SAMPLE_SIZE);
    size_t measureLengthReadOnly = util::min(stepsReadOnly, SAMPLE_SIZE);

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < stepsReadOnly; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            index = tex1Dfetch<uint32_t>(tex, index);
            #endif
        }
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < stepsReadOnly; k++) {
            index = __ldg(&pChaseArrayReadOnly[index]);
        }
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 0) {
        timingResultsTexture[0] += index;

        //second round
        for (uint32_t k = 0; k < measureLengthTexture; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            uint64_t start = __timer();
            index = tex1Dfetch<uint32_t>(tex, index);
            uint64_t end = __timer();

            s_timings1[k] = end - start;
            s_timings1[0] += index; 
            #endif
        }
    }

    __localBarrier(TESTING_THREADS);

    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t s_MemSinkAddr;
    asm volatile(
        "cvta.to.shared.u64 %0, %1;\n\t"  // generic -> shared-space address
        : "=l"(s_MemSinkAddr) // uint64_t
        : "l"(&s_timings2[0]) // __shared__ uint64_t*
    );
    #endif

    if (threadIdx.x == 1) {
        timingResultsReadOnly[0] += index;

        for (uint32_t k = 0; k < measureLengthReadOnly; k++) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            uint64_t start, end;
            const uint32_t* addr = pChaseArrayReadOnly + index;

            asm volatile ( 
                "mov.u64 %0, %%clock64;\n\t" // start = clock()
                "ld.global.nc.u32 %1, [%3];\n\t" // read-only load 
                "st.shared.u32 [%4], %1;\n\t" // sink: force use of loaded value before proceeding
                "mov.u64 %2, %%clock64;\n\t" // end = clock()
                : "=l"(start) // uint64_t
                , "=r"(index) // uint32_t
                , "=l"(end) // uint64_t
                : "l"(addr) // uint32_t*
                , "l"(s_MemSinkAddr) // uint64_t* (shared memory sink)
                : "memory"
            );

            s_timings2[k] = end - start; 
            #endif
        }
    }

    __localBarrier(TESTING_THREADS);

    if (threadIdx.x == 0) {
        for (uint32_t k = 0; k < measureLengthTexture; k++) {
            timingResultsTexture[k] = s_timings1[k];
        }
        timingResultsTexture[0] += index; // dead code elimination prevention
    }

    if (threadIdx.x == 1) {
        for (uint32_t k = 0; k < measureLengthReadOnly; k++) {
            timingResultsReadOnly[k] = s_timings2[k];
        }
        timingResultsReadOnly[0] += index; // dead code elimination prevention
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> textureSharedReadOnlyLauncher(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, size_t readOnlyCacheSizeBytes,size_t readOnlyFetchGranularityBytes) {
    util::hipDeviceReset(); 

    size_t resultBufferLengthTexture = util::min(textureCacheSizeBytes / textureFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 
    size_t resultBufferLengthReadOnly = util::min(readOnlyCacheSizeBytes / readOnlyFetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 

    // Initialize device Arrays
    uint32_t *d_pChaseArrayTexture = util::allocateGPUMemory(util::generatePChaseArray(textureCacheSizeBytes, textureFetchGranularityBytes));
    uint32_t *d_pChaseArrayReadOnly = util::allocateGPUMemory(util::generatePChaseArray(readOnlyCacheSizeBytes, readOnlyFetchGranularityBytes));


    uint32_t *d_timingResultsTexture = util::allocateGPUMemory(resultBufferLengthTexture);
    uint32_t *d_timingResultsReadOnly = util::allocateGPUMemory(resultBufferLengthReadOnly);


    hipTextureObject_t tex = util::createTextureObject(d_pChaseArrayTexture, textureCacheSizeBytes);


    util::hipCheck(hipDeviceSynchronize());
    textureSharedReadOnlyKernel<<<1, util::getMaxThreadsPerBlock()>>>(tex, d_pChaseArrayReadOnly, d_timingResultsTexture, d_timingResultsReadOnly, textureCacheSizeBytes / textureFetchGranularityBytes, readOnlyCacheSizeBytes / readOnlyFetchGranularityBytes);


    std::vector<uint32_t> timingResultBufferTexture = util::copyFromDevice(d_timingResultsTexture, resultBufferLengthTexture);
    std::vector<uint32_t> timingResultBufferReadOnly = util::copyFromDevice(d_timingResultsReadOnly, resultBufferLengthReadOnly);

    timingResultBufferTexture.erase(timingResultBufferTexture.begin());
    timingResultBufferReadOnly.erase(timingResultBufferReadOnly.begin());

    return { timingResultBufferTexture, timingResultBufferReadOnly };
}


namespace benchmark {
    namespace nvidia {
        bool measureTextureAndReadOnlyShared(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, double textureLatency, double textureMissPenalty, size_t readOnlyCacheSizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyLatency, double readOnlyMissPenalty) {
            auto [timingsTexture, timingsReadOnly] = textureSharedReadOnlyLauncher(textureCacheSizeBytes, textureFetchGranularityBytes, readOnlyCacheSizeBytes, readOnlyFetchGranularityBytes);
            
            std::cout << util::average(timingsTexture) << " " << util::average(timingsReadOnly) << std::endl;
            std::cout << textureLatency << " " << textureMissPenalty << " " << readOnlyLatency << " " << readOnlyMissPenalty << std::endl;
            
            return util::average(timingsTexture) - textureLatency > textureMissPenalty / SHARED_THRESHOLD || util::average(timingsReadOnly) - readOnlyLatency > readOnlyMissPenalty / SHARED_THRESHOLD;
        }
    }
}