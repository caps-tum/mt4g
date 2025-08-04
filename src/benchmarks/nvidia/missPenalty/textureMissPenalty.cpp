#include <cstddef>
#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;

__global__ void textureMissPenaltyKernel(hipTextureObject_t tex, uint32_t *timingResults, size_t steps) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    uint32_t measureLength = (uint32_t)util::min(steps, SAMPLE_SIZE);

    // Evict texture cache by loading four times the cache size
    for (uint32_t k = 0; k < steps * 4; k++) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        index = tex1Dfetch<uint32_t>(tex, index);
        #endif
    }

    // index = 0, compiler doesnt know though
    // Second round
    for (uint32_t k = 0; k < measureLength; ++k) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        uint64_t* dst = &s_timings[k];
        asm volatile (
            "{\n\t" // Otherweise duplicate definitions due to unrolling...
                ".reg .u64 t0, t1, delta, saddr, sink64;\n\t"
                ".reg .u32 r0, r1, r2, r3;\n\t"
                "mov.u64 t0, %clock64;\n\t"
                "tex.1d.v4.u32.s32 {r0,r1,r2,r3}, [%2, {%0}];\n\t"
                "cvt.u64.u32 sink64, r0;\n\t"
                "mov.u64 t1, %clock64;\n\t"
                "sub.s64 delta, t1, t0;\n\t"
                "cvta.to.shared.u64 saddr, %1;\n\t"
                "st.shared.u64 [saddr], delta;\n\t"
                "mov.u32 %0, r0;\n\t"  
            "}\n\t"
            : "+r"(index) // uint32_t
            : "l"(dst) // uint64_t* 
            , "l"(tex) // hipTextureObject_t
            : "memory"
        );
        #endif
    }
    
    for (uint32_t k = 0; k < measureLength; k++) {
        timingResults[k] = s_timings[k];
    }

    timingResults[0] = index;
}

std::vector<uint32_t> textureMissPenaltyLauncher(size_t textureCacheSizeBytes, size_t textureCacheLineSizeBytes) {
    util::hipCheck(hipDeviceReset());

    size_t steps = textureCacheSizeBytes / textureCacheLineSizeBytes;
    size_t resultBufferLength = util::min(steps, SAMPLE_SIZE);

    auto initializerArray = util::generatePChaseArray(textureCacheSizeBytes * 4, textureCacheLineSizeBytes);

    // Initialize device Arrays
    uint32_t *d_pChaseArray = util::allocateGPUMemory(initializerArray);
    uint32_t *d_timingResults = util::allocateGPUMemory(resultBufferLength);
    hipTextureObject_t tex = util::createTextureObject(d_pChaseArray, textureCacheSizeBytes * 4);

    util::hipCheck(hipDeviceSynchronize());
    textureMissPenaltyKernel<<<1, 1>>>(tex, d_timingResults, steps);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, resultBufferLength);

    timingResultBuffer.erase(timingResultBuffer.begin());


    return timingResultBuffer;
}


namespace benchmark {
    namespace nvidia {
        double measureTextureMissPenalty(size_t textureCacheSizeBytes, size_t textureCacheLineSizeBytes, double textureLatency) {
            auto timings = textureMissPenaltyLauncher(textureCacheSizeBytes, textureCacheLineSizeBytes);

            return std::abs(util::average(timings) - textureLatency);
        }
    }
}