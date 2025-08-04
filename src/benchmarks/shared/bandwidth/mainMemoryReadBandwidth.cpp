#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR;// Factor
static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void mainMemoryReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0}; 

    for (size_t i = tid; i < n; i += stride) {
        uint32v4 loaded; 
        
        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(loaded.x) // int
            , "=r"(loaded.y) // int
            , "=r"(loaded.z) // int
            , "=r"(loaded.w) // int
            : "l"(src + i) // uint32v4*
        );
        #endif

        #ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "flat_load_dwordx4 %0, %1, " GLC "\n" 
            : "=v"(loaded) // uint32v4
            : "v"(src + i) // uint32v4*
            :
        );
        #endif

        // XOR is efficient
        dummy.x ^= loaded.x;
    }

    dst[tid] = dummy; // prevent dead code elimination
}

double mainMemoryReadBandwidthLauncher(size_t arraySizeBytes) { 
    util::hipCheck(hipDeviceReset()); 

    uint32_t maxThreadsPerBlock = util::getMaxThreadsPerBlock(); // Allows the scheduler to use extensive latency hiding
    uint32_t maxBlocks = 2048; // TODO: Find heuristic (funnily, departure delay might be interesting for that)

    // Initialize device Arrays
    // sizeof(uint32v4) = 16 bytes -> allows us to load 4 integers with one instruction -> probability 
    // of the bandwidth being limited by the memory bandwidth rather than compute is considerably higher
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(maxBlocks * maxThreadsPerBlock); // total threads
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    mainMemoryReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, arraySizeBytes / sizeof(uint32v4));
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end);
}

namespace benchmark {
    double measureMainMemoryReadBandwidth(size_t mainMemorySizeBytes) {
        size_t testSizeBytes = mainMemorySizeBytes / SIZE_DOWN; // Divide by SIZE_DOWN to avoid too large memory allocations
        double testSizeGiB = testSizeBytes / (1 * GiB); // Convert to GiB

        std::vector<double> results(ROUNDS);
        for (uint32_t i = 0; i < ROUNDS; ++i) {
            results[i] = mainMemoryReadBandwidthLauncher(testSizeBytes) / MS_PER_SECOND;
        }
        
        return testSizeGiB / util::average(results); 
    }
}