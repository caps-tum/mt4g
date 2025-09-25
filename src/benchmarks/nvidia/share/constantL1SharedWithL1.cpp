#include <cstddef>
#include <hip/hip_runtime.h>

#include "const/constArray16384.hpp"
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE;
static constexpr auto ROUNDS = 10;

__global__ void constantL1SharedL1Kernel(uint32_t* pChaseArrayL1, uint32_t *timingResultsConstantL1, uint32_t *timingResultsL1, size_t stepsConstantL1, size_t stepsL1, size_t constantL1Stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return; // Ensure only two threads are used
    uint32_t index = 0;

    __shared__ uint64_t s_timings1[SAMPLE_SIZE];
    __shared__ uint64_t s_timings2[SAMPLE_SIZE];

    size_t measureLengthConstantL1 = util::min(stepsConstantL1, SAMPLE_SIZE);
    size_t measureLengthL1 = util::min(stepsL1, SAMPLE_SIZE);

    for (uint32_t k = 0; k < stepsConstantL1; k++) {
        index = arr16384AscStride0[index] + constantL1Stride;
    }

    timingResultsConstantL1[0] += index;
    s_timings1[0] += index;

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        "mov.u32 %0, 0;\n\t"
        : "=r"(index)
        : 
        : "memory"
    );
    #endif
    
    for (uint32_t k = 0; k < stepsL1; k++) {
        index = __allowL1Read(pChaseArrayL1, index);
    }

    // second round
    // No Fine Grained Measurements because it leads to unexpected behavior on some GPUs. TODO: Find out why and fix. 
    // Works nonetheless because measure just calculates the avg anyways
    uint64_t start = clock();
    for (uint32_t k = 1; k < measureLengthConstantL1; k++) {
        index = arr16384AscStride0[index] + constantL1Stride;
    }
    uint64_t end = clock();
    s_timings1[0] += index;
    s_timings1[1] = end - start;

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %0;\n\t"
        :
        : "l"(s_timings2)
    );
    #endif

    timingResultsL1[0] += index;

    #ifdef __HIP_PLATFORM_NVIDIA__
    asm volatile(
        "mov.u32 %0, 0;\n\t"
        : "=r"(index)
        : 
        : "memory"
    );
    #endif
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
        s_timings2[k] = end - start;
        s_timings2[0] += index;
        #endif
    }

    for (uint32_t k = 1; k < measureLengthConstantL1; k++) {
        timingResultsConstantL1[k] = s_timings1[k];
    }

    for (uint32_t k = 1; k < measureLengthL1; k++) {
        timingResultsL1[k] = s_timings2[k];
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> constantL1SharedL1Launcher(size_t constantL1CacheSizeBytes, size_t constantL1FetchGranularityBytes, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes) {
    util::hipDeviceReset(); 

    size_t resultBufferLengthConstantL1 = util::min(constantL1CacheSizeBytes / constantL1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 
    size_t resultBufferLengthL1 = util::min(l1CacheSizeBytes / l1FetchGranularityBytes, SAMPLE_SIZE / sizeof(uint32_t)); 

    // Initialize device Arrays
    uint32_t *d_pChaseArrayL1 = util::allocateGPUMemory(util::generatePChaseArray(l1CacheSizeBytes, l1FetchGranularityBytes));

    uint32_t *d_timingResultsConstantL1 = util::allocateGPUMemory(resultBufferLengthConstantL1);
    uint32_t *d_timingResultsL1 = util::allocateGPUMemory(resultBufferLengthL1);


    util::hipCheck(hipDeviceSynchronize());
    constantL1SharedL1Kernel<<<1, util::getMaxThreadsPerBlock()>>>(d_pChaseArrayL1, d_timingResultsConstantL1, d_timingResultsL1, constantL1CacheSizeBytes / constantL1FetchGranularityBytes, l1CacheSizeBytes / l1FetchGranularityBytes, constantL1FetchGranularityBytes / sizeof(uint32_t));


    std::vector<uint32_t> timingResultBufferConstantL1 = util::copyFromDevice(d_timingResultsConstantL1, resultBufferLengthConstantL1);
    std::vector<uint32_t> timingResultBufferL1 = util::copyFromDevice(d_timingResultsL1, resultBufferLengthL1);

    timingResultBufferConstantL1.erase(timingResultBufferConstantL1.begin());
    timingResultBufferL1.erase(timingResultBufferL1.begin());

    return { timingResultBufferConstantL1, timingResultBufferL1 };
}


namespace benchmark {
    namespace nvidia {
        bool measureConstantL1AndL1Shared(size_t constantL1CacheSizeBytes, size_t constantL1FetchGranularityBytes, double constantL1Latency, double constantL1MissPenalty, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty) {
            auto [timingsConstantL1, timingsL1] = constantL1SharedL1Launcher(constantL1CacheSizeBytes, constantL1FetchGranularityBytes, l1CacheSizeBytes, l1FetchGranularityBytes);
            
            //util::printVector(timingsConstantL1);
            //util::printVector(timingsL1);

            //std::cout << "Constant L1 Latency: " << util::average(timingsConstantL1) << ", L1 Latency: " << util::average(timingsL1) << std::endl;
            //std::cout << constantL1Latency << " " << constantL1MissPenalty << " " << l1Latency << " " << l1MissPenalty << std::endl;

            bool shared = true;

            // repeat ROUNDS times because its not too reliable in its current form. We expect it to not act as shared. Higher latencies in the C1 part are probably because of scheduling details
            for (uint32_t i = 0; i < ROUNDS; i++) {
                auto [timingsConstantL1Round, timingsL1Round] = constantL1SharedL1Launcher(constantL1CacheSizeBytes, constantL1FetchGranularityBytes, l1CacheSizeBytes, l1FetchGranularityBytes);
                shared &= util::average(timingsConstantL1Round) - constantL1Latency > constantL1MissPenalty / SHARED_THRESHOLD && util::average(timingsL1Round) - l1Latency > l1MissPenalty / SHARED_THRESHOLD;
            }

            return shared;
        }
    }
}