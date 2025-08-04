#include <cstddef>
#include <vector>
#include <map>
#include <tuple>
#include <set>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

static constexpr auto MEASURE_SIZE = 1024;

static constexpr auto THREADBLOCKS = 2;

__global__ void cuShareScalarL1Kernel(uint32_t *pChaseArrayBaseCU, uint32_t *pChaseArrayTestCU, uint32_t *timingResultsBaseCU, uint32_t *timingResultsTestCU, size_t steps, uint32_t baseCU, uint32_t testCU) {
    __shared__ uint64_t s_timingResultsBaseCU[MEASURE_SIZE];
    __shared__ uint64_t s_timingResultsTestCU[MEASURE_SIZE];

    uint32_t index = 0;
    uint32_t measureLength = util::min(steps, MEASURE_SIZE);

    uint32_t currentCUId = __getPhysicalCUId(); // Retrieves the physical ID

    timingResultsBaseCU[0] = 13371337; // Marker values to check if both threadblocks run on different CUs
    timingResultsTestCU[0] = 13371337;

    __globalBarrier(THREADBLOCKS);

    if (currentCUId == baseCU) {
        timingResultsBaseCU[0] = currentCUId;
    } else if (currentCUId == testCU) {
        timingResultsTestCU[0] = currentCUId;
    } else {
        return; // Something went horribly wrong (stream assignment did not work)
    }

    __globalBarrier(THREADBLOCKS);

    if (timingResultsBaseCU[0] == 13371337 || timingResultsTestCU[0] == 13371337) {
        // This means that the CUs are not different, so we cannot measure anything
        return;
    }
    
    __globalBarrier(THREADBLOCKS);

    // Now we know that both thread blocks run on the correct CU, testing can continue as normal

    // Let the base CU load the first steps values
    if (currentCUId == baseCU) {
        for (uint32_t k = 0; k < steps; k++) {
            #ifdef __HIP_PLATFORM_AMD__
            uint32_t *addr = pChaseArrayBaseCU + index;

            asm volatile(
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                "s_load_dword %0, %1, 0\n\t" // index = *addr;

                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                // Last syncs
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                : "+s"(index) //uint32_t
                , "+s"(addr) // uint32_t*
                :
                : "memory"
            );
            #endif
        }

        timingResultsBaseCU[0] = index >> util::min(steps, 32);
        index = 0;
    }

    __globalBarrier(THREADBLOCKS); // Ensure both threadblocks are here

    // If the CUs share the same cache physically this will evict all values loaded before
    if (currentCUId == testCU) {
        for (uint32_t k = 0; k < steps; k++) {
            #ifdef __HIP_PLATFORM_AMD__
            uint32_t *addr = pChaseArrayTestCU + index;

            asm volatile(
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                "s_load_dword %0, %1, 0\n\t" // index = *addr;

                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                // Last syncs
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                : "+s"(index) //uint32_t
                , "+s"(addr) // uint32_t*
                :
                : "memory"
            );
            #endif
        }

        timingResultsTestCU[0] = index >> util::min(steps, 32);
        index = 0;
    }

    __globalBarrier(THREADBLOCKS);

    if (currentCUId == baseCU) {
        //second round
        for (uint32_t k = 0; k < measureLength; ++k) {
            #ifdef __HIP_PLATFORM_AMD__
            uint64_t start, end;
            uint32_t *addr = pChaseArrayBaseCU + index;

            asm volatile(
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                "s_memtime %0\n\t" // start = clock();

                "s_load_dword %2, %3, 0\n\t" // index = *addr;

                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                "s_memtime %1\n\t" // end = clock();

                // Last syncs
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                : "+s"(start) // uint64_t
                , "+s"(end) // uint64_t
                , "+s"(index) //uint32_t
                , "+s"(addr) // uint32_t*
                :
                : "memory"
            );

            s_timingResultsBaseCU[k] = end - start;
            #endif
        }
        
        timingResultsBaseCU[0] += index >> util::min(steps, 32);
    }

    __globalBarrier(THREADBLOCKS);

    if (currentCUId == testCU) {
        //second round
        for (uint32_t k = 0; k < measureLength; ++k) {
            #ifdef __HIP_PLATFORM_AMD__
            uint64_t start, end;
            uint32_t *addr = pChaseArrayTestCU + index;

            asm volatile(
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                "s_memtime %0\n\t" // start = clock();

                "s_load_dword %2, %3, 0\n\t" // index = *addr;

                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                "s_memtime %1\n\t" // end = clock();

                // Last syncs
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                : "+s"(start) // uint64_t
                , "+s"(end) // uint64_t
                , "+s"(index) //uint32_t
                , "+s"(addr) // uint32_t*
                :
                : "memory"
            );

            s_timingResultsTestCU[k] = end - start;
            #endif
        }

        timingResultsTestCU[0] += index >> util::min(steps, 32);
    }

    __globalBarrier(THREADBLOCKS);

    if (currentCUId == baseCU) {
        for (uint32_t k = 1; k < measureLength; k++) {
            timingResultsBaseCU[k] = s_timingResultsBaseCU[k];
        }
    }

    if (currentCUId == testCU) {
        for (uint32_t k = 1; k < measureLength; k++) {
            timingResultsTestCU[k] = s_timingResultsTestCU[k];
        }
    }
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> cuSharescalarL1Launcher(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes, uint32_t baseCU, uint32_t testCU, std::vector<uint32_t> physicalCUIdLUT) {
    util::hipCheck(hipDeviceReset()); 

    std::vector<uint32_t> initializerArray = util::generatePChaseArray(scalarL1SizeBytes, scalarL1FetchGranularityBytes);

    uint32_t *d_pChaseArrayBaseCU = util::allocateGPUMemory(initializerArray);
    uint32_t *d_pChaseArrayTestCU = util::allocateGPUMemory(initializerArray);

    size_t resultBufferLength = util::min(scalarL1SizeBytes / scalarL1FetchGranularityBytes, MEASURE_SIZE);

    uint32_t *d_timingResultsBaseCU = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCU = util::allocateGPUMemory(resultBufferLength);

    util::hipCheck(hipMemset(d_timingResultsBaseCU, 0, resultBufferLength * sizeof(uint32_t)));
    util::hipCheck(hipMemset(d_timingResultsTestCU, 0, resultBufferLength * sizeof(uint32_t)));


    std::vector<uint32_t> baseCUTimingResultsBuffer;
    std::vector<uint32_t> testCUTimingResultsBuffer;

    do {
        
        util::hipCheck(hipDeviceSynchronize());
        cuShareScalarL1Kernel<<<THREADBLOCKS, 1, 0, util::createStreamForCUs({ baseCU, testCU })>>>(d_pChaseArrayBaseCU, d_pChaseArrayTestCU, d_timingResultsBaseCU, d_timingResultsTestCU, scalarL1SizeBytes / scalarL1FetchGranularityBytes, physicalCUIdLUT[baseCU], physicalCUIdLUT[testCU]);

        baseCUTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCU, resultBufferLength);
        testCUTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCU, resultBufferLength);

        baseCUTimingResultsBuffer.erase(baseCUTimingResultsBuffer.begin());
        testCUTimingResultsBuffer.erase(testCUTimingResultsBuffer.begin());

    } while (((baseCUTimingResultsBuffer[0] == 0) && (testCUTimingResultsBuffer[0] != 0)) || 
             ((baseCUTimingResultsBuffer[0] != 0) && (testCUTimingResultsBuffer[0] == 0)));  
             // The scheduler does not guarantee that both CUs will be used, so we may have to retry if one of the CUs did not run the kernel

    return { baseCUTimingResultsBuffer, testCUTimingResultsBuffer };
}

namespace benchmark {
    namespace amd {
        std::set<std::set<uint32_t>> measureCuShareScalarL1(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes) {
            auto logicalToPhysicalCUs = util::getLogicalToPhysicalCUsLUT();

            DisjointSet dsu;

            std::set<uint32_t> skipLUT;

            for (uint32_t k = 0; k < util::getNumberOfComputeUnits(); ++k) {
                std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCUToTimingResults;

                for (uint32_t i = 0; i < util::getNumberOfComputeUnits(); ++i) { 
                    if (k == i || skipLUT.contains(logicalToPhysicalCUs.at(i))) continue;
                    testCUToTimingResults[logicalToPhysicalCUs[i]] = cuSharescalarL1Launcher(scalarL1SizeBytes, scalarL1FetchGranularityBytes, k, i, logicalToPhysicalCUs);
                }

                auto mostLikelyPartners = util::detectShareChangePoint(testCUToTimingResults);
                
                util::printVector(mostLikelyPartners);

                if (!mostLikelyPartners.empty()) {
                    for (auto partnerId : mostLikelyPartners) {
                        dsu.unite(logicalToPhysicalCUs[k], partnerId);
                    }
                    skipLUT.insert(logicalToPhysicalCUs[k]);
                } else {
                    dsu.add(logicalToPhysicalCUs[k]);
                    skipLUT.insert(logicalToPhysicalCUs[k]);
                }
            }

            return dsu.getEquivalenceClasses();
        } 
    }
}
