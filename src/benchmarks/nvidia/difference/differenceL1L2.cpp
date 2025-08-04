#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cmath>

static constexpr auto SAMPLE_SIZE = 100;// 100 Loads should suffice to rule out random flukes

__global__ void l1DifferKernel([[maybe_unused]] uint32_t *pChaseArray, uint32_t *timingResults) {
    __shared__ uint64_t s_timings[SAMPLE_SIZE];
    [[maybe_unused]] __shared__ uint32_t s_index_l1[SAMPLE_SIZE];

    #ifdef __HIP_PLATFORM_NVIDIA__ // Hide from hipcc
    uint32_t index = 0;
    uint32_t *ptr;

    // First round
    for (int32_t k = 0; k < SAMPLE_SIZE; k++) {
        ptr = pChaseArray + index;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(index) : "l"(ptr) : "memory");
    }

    // Second round
    asm volatile(" .reg .u64 smem_ptr64;\n\t"
                " cvta.to.shared.u64 smem_ptr64, %0;\n\t" :: "l"(s_index_l1));
    for (int32_t k = 0; k < SAMPLE_SIZE; k++) {
        uint32_t start, end;
        ptr = pChaseArray + index;
        asm volatile ("mov.u32 %0, %%clock;\n\t"
                      "ld.global.ca.u32 %1, [%3];\n\t"
                      "st.shared.u32 [smem_ptr64], %1;"
                      "mov.u32 %2, %%clock;\n\t"
                      "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start), "=r"(index), "=r"(end) : "l"(ptr) : "memory");
        s_timings[k] = end - start;
    }

    #endif

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        //indexL1[k]= s_index_l1[k];
        timingResults[k] = s_timings[k];
    }
}


//__attribute__((optimize("O0"), noinline))
__global__ void l2DifferKernel([[maybe_unused]] uint32_t *pChaseArray, uint32_t *timingResults) { 
    __shared__ uint64_t s_timings[SAMPLE_SIZE];
    [[maybe_unused]] __shared__ uint32_t s_index_l2[SAMPLE_SIZE];


    #ifdef __HIP_PLATFORM_NVIDIA__

    uint32_t index = 0;
    uint32_t* ptr;

    // First round
    for (int32_t k = 0; k < SAMPLE_SIZE; k++) {
        ptr = pChaseArray + index;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(index) : "l"(ptr) : "memory");
    }

    // Second round
    for (int32_t k = 0; k < SAMPLE_SIZE; k++) {
        uint32_t start, end;
        ptr = pChaseArray + index;
        // start = clock();
        asm volatile ("mov.u32 %0, %%clock;\n\t"
                      "ld.global.cg.u32 %1, [%2];\n\t" : "=r"(start), "=r"(index) : "l"(ptr) : "memory");
        s_index_l2[k] = index;
        asm volatile ("mov.u32 %0, %%clock;\n\t" : "=r"(end));
        s_timings[k] = end - start;
    }

    #endif

    for (uint32_t k = 0; k < SAMPLE_SIZE; k++) {
        timingResults[k] = s_timings[k] + (s_index_l2[k] & 0x1); // Does not change the difference significantly (+-1)
    }
}

namespace benchmark {
    namespace nvidia {
        bool isL1UsedForGlobalLoads(double tolerance) {
            util::hipCheck(hipDeviceReset());


            uint32_t *d_pChaseArray = util::allocateGPUMemory(util::generatePChaseArray(SAMPLE_SIZE, sizeof(uint32_t)));
            uint32_t *d_timingResultBuffer = util::allocateGPUMemory(SAMPLE_SIZE);
                        
            util::hipCheck(hipDeviceSynchronize());
            l2DifferKernel<<<1, 1>>>(d_pChaseArray, d_timingResultBuffer); 
            std::vector<uint32_t> timingResultBufferL2 = util::copyFromDevice(d_timingResultBuffer, SAMPLE_SIZE);


            //util::hipCheck(hipDeviceReset()); // Flush caches

            
            util::hipCheck(hipDeviceSynchronize());
            l1DifferKernel<<<1, 1>>>(d_pChaseArray, d_timingResultBuffer); 
            std::vector<uint32_t> timingResultBufferL1 = util::copyFromDevice(d_timingResultBuffer, SAMPLE_SIZE);
            

            util::hipCheck(hipDeviceReset());

            std::cout << "L1 " << util::average(timingResultBufferL1) << " L2 " << util::average(timingResultBufferL2) << std::endl;

            //util::printVector(timingResultBufferL2);

            return std::abs(util::average(timingResultBufferL1) - util::average(timingResultBufferL2)) >= tolerance;
        }
    }
}