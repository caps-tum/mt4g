/* 

THIS BENCHMARK MAKES NO SENSE AS EACH CU ONLY HAS ONE SALU

#include <cstddef>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <iterator>

#include "const/constArray16384.hpp"
#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = DEFAULT_SAMPLE_SIZE;// Values, upper limit

__global__ void scalarL1AmountKernel(uint32_t *timingResultsMatrix, size_t steps, uint32_t stride) {
    uint32_t threadCount = gridDim.x * blockDim.x;
    size_t measureLength = util::min(steps, MEASURE_SIZE);

    __shared__ uint64_t s_timingResults[MEASURE_SIZE];

    for (uint32_t currentIdx = 0; currentIdx < threadCount; ++currentIdx) {
        uint32_t index = 0;
        
        if (threadIdx.x == currentIdx) {
            index = CONST_ARRAY_SIZE - steps * stride; // testCore should load other values than baseCore. 
        }

        // If the threads share the same cache physically those values will be evicted in the next loop
        if (threadIdx.x == currentIdx) {
            for (uint32_t k = 0; k < steps; k++) {
                index = arr16384AscStride0[index] + stride;
            }
            s_timingResults[0] += index >> util::min(steps, 32);
            index = CONST_ARRAY_SIZE - steps * stride;
        }

        __syncthreads();
        // Let the base Core load the first steps-values
        if (threadIdx.x == 0) {
            for (uint32_t k = 0; k < steps; k++) {
                index = arr16384AscStride0[index] + stride;
            }

            s_timingResults[0] += index >> util::min(steps, 32);
            index = 0;
        }

        __syncthreads();

        if (threadIdx.x == currentIdx) {
            for (uint32_t k = 0; k < measureLength; k++) {
                uint64_t addr = reinterpret_cast<uint64_t>(arr16384AscStride0) + static_cast<uint64_t>(index) * sizeof(uint32_t);

                V_TO_SGPR64(s10, s11, addr);
                V_TO_SGPR32(s12, stride);
                V_TO_SGPR32(s13, index);

                asm volatile(
                    "s_waitcnt lgkmcnt(0)\n\t"
                    "s_waitcnt vmcnt(0)\n\t"

                    "s_memtime s[14:15]\n\t" // start = clock()

                    "s_load_dword s16, s[10:11], 0x0\n\t" // loaded = arr16384AscStride0[0]

                    "s_waitcnt lgkmcnt(0)\n\t"
                    "s_waitcnt vmcnt(0)\n\t"

                    "s_memtime s[18:19]\n\t" // end = clock()

                    "s_add_u32 s13, s16, s12\n\t" // index = loaded + stride

                    "s_waitcnt lgkmcnt(0)\n\t"
                    "s_waitcnt vmcnt(0)\n\t"
                    :
                    : 
                    : "memory", 
                    "s14", "s15", // = start
                    "s16", // = loaded
                    "s18", "s19" // = end
                );

                uint64_t start;
                SGPR_TO_VAR64(start, s14, s15);
                uint64_t end;
                SGPR_TO_VAR64(end, s18, s19);
                SGPR_TO_VAR32(index, s16);

                s_timingResults[k] = end - start;
            }
            s_timingResults[0] += index >> util::min(steps, 32);
        }

        __syncthreads();

        if (threadIdx.x == currentIdx) {
            for (uint32_t k = 0; k < measureLength; k++) {
                timingResultsMatrix[currentIdx * measureLength + k] = s_timingResults[k];
            }
        }
    }
}


std::map<uint32_t, std::vector<uint32_t>> scalarL1AmountLauncher(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes) {
    util::hipCheck(hipDeviceReset()); 

    scalarL1SizeBytes = util::min(scalarL1SizeBytes, CONST_ARRAY_SIZE); // Cap at CONST_ARRAY_SIZE, otherwise the benchmark will access illegal addresses and returnt trash values

    size_t steps = scalarL1SizeBytes / scalarL1FetchGranularityBytes; 
    size_t resultBufferLength = util::min(steps, MEASURE_SIZE) * util::getWarpSize();

    uint32_t *d_timingResultsMatrix = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipDeviceSynchronize());
    scalarL1AmountKernel<<<1, util::getWarpSize()>>>(d_timingResultsMatrix, steps, scalarL1FetchGranularityBytes / sizeof(uint32_t));

    std::vector<uint32_t> baseCoreTimingResultsBuffer = util::copyFromDevice(d_timingResultsMatrix, resultBufferLength);

    std::map<uint32_t, std::vector<uint32_t>> result;


    for (uint32_t i = 0; i < util::getWarpSize(); ++i) {
        result[i] = util::getSlice(baseCoreTimingResultsBuffer, util::getWarpSize(), i);
    }

    return result;
}

namespace benchmark {
    namespace amd {
        uint32_t measureScalarL1Amount(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes) {
            if (scalarL1SizeBytes > CONST_ARRAY_SIZE) {
                std::cout << "Scalar L1 is too large to be benchmarked correctly." << std::endl;
            }

            auto threadToAccessTimes = scalarL1AmountLauncher(scalarL1SizeBytes, scalarL1FetchGranularityBytes);

            util::printMap(threadToAccessTimes);

            return 0;
        }
    }
}
*/