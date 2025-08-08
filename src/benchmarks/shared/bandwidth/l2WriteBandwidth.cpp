#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>


static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void l2WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n) {
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    uint32v4 dummy = { tid, tid + 1, tid + 2, tid + 3 }; 

    for (size_t j = 0; j < blockDim.x; ++j) {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;
        
        for (size_t i = tid; i < n; i += stride) {

            #ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "st.global.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(dst + i) // uint32v4*
                , "r"(dummy.x) // int
                , "r"(dummy.y) // int
                , "r"(dummy.z) // int
                , "r"(dummy.w) // int
            );
            #endif

            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_store_dwordx4 %0, %1\n"
                :
                : "v"(dst + i) // uint32v4*
                , "v"(dummy) // uint32v4
                : "memory"
            );
            #endif

        }
    }
}


double l2WriteBandwidthLauncher(size_t arraySizeBytes) { 
    util::hipDeviceReset(); 

    // Calculate number of blocks and threads
    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU()); 
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

   
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));

    // warm up L2
    l2WriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4));
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l2WriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4));
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());
    
    return util::getElapsedTimeMs(start, end) / maxThreadsPerBlock; // Diff between end and start is blockDim.x * TimeItTakesToLoadArraySizeBytes
}

namespace benchmark {
    double measureL2WriteBandwidth(size_t l2SizeBytes) {
        double testSizeGiB = (double)l2SizeBytes / (1 * GiB); // Convert to GiB

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) {
            results[i] = l2WriteBandwidthLauncher(l2SizeBytes) / MS_PER_SECOND;
        }

        return testSizeGiB / util::average(results);
    }
}