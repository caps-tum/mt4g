    /*
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>


static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds
static constexpr auto SCRATCH_SIZE = DEFAULT_SCRATCH_SIZE;

// NOT WORKING
// Wrong approach. Needs to run bandwidth benchmark multiple times and find optimal thread block size and thread count
__global__ void departureDelayKernel(int4* __restrict__ read, int4* __restrict__ write, uint64_t *timingResults) {

    int4 buf1 = {0, 0, 0, 0}; 
    int4 buf2 = {0, 0, 0, 0}; 
    int4 buf3 = {0, 0, 0, 0}; 
    int4 buf4 = {0, 0, 0, 0}; 
    int4 buf5 = {0, 0, 0, 0}; 
    int4 buf6 = {0, 0, 0, 0}; 
    int4 buf7 = {0, 0, 0, 0}; 
    int4 buf8 = {0, 0, 0, 0}; 
    int4 *read1 = read + 0;
    int4 *read2 = read + 1;
    int4 *read3 = read + 2;
    int4 *read4 = read + 3; 
    int4 *read5 = read + 4; 
    int4 *read6 = read + 5; 
    int4 *read7 = read + 6;
    int4 *read8 = read + 7;  
    #ifdef __HIP_PLATFORM_NVIDIA__
    uint32_t startRead, endRead, startWrite, endWrite;
    __syncthreads();
    asm volatile (
        "barrier.sync 0;\n\t" // Ensure all threads have completed their loads before proceeding
        "mov.u32 %0, %%clock;\n\t"
        "ld.global.v4.u32 {%1,%2,%3,%4}, [%34];\n\t"
        "ld.global.v4.u32 {%5,%6,%7,%8}, [%35];\n\t"
        "ld.global.v4.u32 {%9,%10,%11,%12}, [%36];\n\t"
        "ld.global.v4.u32 {%13,%14,%15,%16}, [%37];\n\t"
        "ld.global.v4.u32 {%17,%18,%19,%20}, [%38];\n\t"
        "ld.global.v4.u32 {%21,%22,%23,%24}, [%39];\n\t"
        "ld.global.v4.u32 {%25,%26,%27,%28}, [%40];\n\t"
        "ld.global.v4.u32 {%29,%30,%31,%32}, [%41];\n\t"
        "mov.u32 %33, %%clock;\n\t"
        : "=r"(startRead) // 0
        , "=r"(buf1.x), "=r"(buf1.y), "=r"(buf1.z), "=r"(buf1.w)
        , "=r"(buf2.x), "=r"(buf2.y), "=r"(buf2.z), "=r"(buf2.w)
        , "=r"(buf3.x), "=r"(buf3.y), "=r"(buf3.z), "=r"(buf3.w)
        , "=r"(buf4.x), "=r"(buf4.y), "=r"(buf4.z), "=r"(buf4.w)
        , "=r"(buf5.x), "=r"(buf5.y), "=r"(buf5.z), "=r"(buf5.w)
        , "=r"(buf6.x), "=r"(buf6.y), "=r"(buf6.z), "=r"(buf6.w)
        , "=r"(buf7.x), "=r"(buf7.y), "=r"(buf7.z), "=r"(buf7.w)
        , "=r"(buf8.x), "=r"(buf8.y), "=r"(buf8.z), "=r"(buf8.w)
        , "=r"(endRead) // 33
        : "l"(read1) 
        , "l"(read2) 
        , "l"(read3) 
        , "l"(read4) 
        , "l"(read5) 
        , "l"(read6) 
        , "l"(read7)
        , "l"(read8)  
        : "memory"
    );
    *timingResults = (endRead - startRead) / 8;
    #endif

    #ifdef __HIP_PLATFORM_AMD__
    #endif
}

std::tuple<double, double> departureDelayLauncher() { 
    util::hipDeviceReset(); 

    std::vector<double> results;
    results.reserve(ROUNDS);

    int4 *d_read = util::allocateGPUMemory<int4>(8);
    int4 *d_write = util::allocateGPUMemory<int4>(1);
    uint64_t *d_timingResults = util::allocateGPUMemory<uint64_t>(2); // 0 = readCycles, 1 = writeCycles

    util::hipCheck(hipDeviceSynchronize());
    departureDelayKernel<<<1, 1024>>>(d_read, d_write, d_timingResults);

    std::vector<uint64_t> timingResultBuffer = util::copyFromDevice(d_timingResults, 1);

    util::printVector(timingResultBuffer);

    return {0,0 };
}

namespace benchmark {
    std::tuple<double, double> measureDepartureDelay() {
        return departureDelayLauncher();
    }
}
    */