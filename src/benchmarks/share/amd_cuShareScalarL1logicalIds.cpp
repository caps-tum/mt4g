#include <cstddef>
#include <vector>
#include <map>
#include <tuple>
#include <set>
#include <iostream>

#include <hip/hip_runtime.h>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

/* not validated

static constexpr auto MEASURE_SIZE = 1024;

__device__ __forceinline__ uint64_t readMemtime()
{
    #ifdef __HIP_PLATFORM_AMD__
    uint64_t t;
    asm volatile(
        "s_memtime %0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=s"(t));
    return t;
    #else
    return static_cast<uint64_t>(clock64());
    #endif
}

struct alignas(8) Barrier2 
{
    uint32_t count;
    uint32_t sense;
};

__device__ __forceinline__ void barrier2_wait(Barrier2* b) 
{
    uint32_t local = 1u - __atomic_load_n(&b->sense, __ATOMIC_ACQUIRE);

    uint32_t ticket = atomicAdd(reinterpret_cast<unsigned int*>(&b->count), 1u);

    if (ticket == 1u) 
    {
        __threadfence();

        __atomic_store_n(&b->count, 0u, __ATOMIC_RELAXED);
        __atomic_store_n(&b->sense, local, __ATOMIC_RELEASE);
    } 
    else 
    {
        while (__atomic_load_n(&b->sense, __ATOMIC_ACQUIRE) != local) 
        {
            #ifdef __HIP_PLATFORM_AMD__
            __builtin_amdgcn_s_sleep(1);
            #endif
        }
        __threadfence();
    }
}

__global__ void cuShareScalarL1LogicalKernel(uint32_t* pChaseArray, uint32_t* timingResults, Barrier2* b, size_t steps, bool isBaseCU) 
{
    __shared__ uint64_t s_timingResults[MEASURE_SIZE];

    for (int i = 0; i < MEASURE_SIZE; ++i)
    {
        s_timingResults[i] = 0;
    }

    uint32_t index = 0;
    uint32_t measureLength = util::min(steps, MEASURE_SIZE);

    barrier2_wait(b);

    // Let the base CU load the first steps values
    if (isBaseCU) 
    {
        for (uint32_t k = 0; k < steps; k++) 
        {
            #ifdef __HIP_PLATFORM_AMD__
            uint32_t *addr = pChaseArray + index;

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

        timingResults[0] = index >> util::min(steps, 32);
        index = 0;
    }

    barrier2_wait(b);

    // If the CUs share the same cache physically this will evict all values loaded before
    if (!isBaseCU) 
    {
        for (uint32_t k = 0; k < steps; k++) 
        {
            #ifdef __HIP_PLATFORM_AMD__
            uint32_t *addr = pChaseArray + index;

            asm volatile(
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                "s_load_dword %0, %1, 0\n\t" // index = *addr;

                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                // Last syncs
                "s_waitcnt lgkmcnt(0)\n\t"
                "s_waitcnt vmcnt(0)\n\t"

                : "=s"(index) //uint32_t
                : "s"(addr) // uint32_t*
                : "memory"
            );
            #endif
        }

        timingResults[0] = index >> util::min(steps, 32);
        index = 0;
    }

    barrier2_wait(b);

    if (isBaseCU) 
    {
        //second round
        for (uint32_t k = 0; k < measureLength; ++k) 
        {
            #ifdef __HIP_PLATFORM_AMD__
            uint64_t start, end;
            uint32_t *addr = pChaseArray + index;

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

                : "=s"(start) // uint64_t
                , "=s"(end) // uint64_t
                , "=s"(index) //uint32_t
                : "s"(addr) // uint32_t*
                : "memory"
            );

            s_timingResults[k] = end - start;
            #endif
        }

        timingResults[0] += index >> util::min(steps, 32);
    }

    barrier2_wait(b);

    if (!isBaseCU) 
    {
        //second round
        for (uint32_t k = 0; k < measureLength; ++k) 
        {
            #ifdef __HIP_PLATFORM_AMD__
            uint64_t start, end;
            uint32_t *addr = pChaseArray + index;

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

                : "=s"(start) // uint64_t
                , "=s"(end) // uint64_t
                , "=s"(index) //uint32_t
                : "s"(addr) // uint32_t*
                : "memory"
            );

            s_timingResults[k] = end - start;
            #endif
        }


        timingResults[0] += index >> util::min(steps, 32);
    }

    barrier2_wait(b);

    for (uint32_t k = 1; k < measureLength; ++k) 
    {
        timingResults[k] = s_timingResults[1];
    }
}

std::optional<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> cuSharescalarL1LogicalLauncher(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes, uint32_t baseCU, uint32_t testCU) 
{
    std::vector<uint32_t> initializerArray = util::generatePChaseArray(scalarL1SizeBytes, scalarL1FetchGranularityBytes);

    const size_t steps = scalarL1SizeBytes / scalarL1FetchGranularityBytes;
    
    uint32_t *d_pChaseArrayBaseCU = util::allocateGPUMemory(initializerArray);
    uint32_t *d_pChaseArrayTestCU = util::allocateGPUMemory(initializerArray);

    const size_t resultBufferLength = util::min(steps, static_cast<size_t>(MEASURE_SIZE));

    uint32_t *d_timingResultsBaseCU = util::allocateGPUMemory(resultBufferLength);
    uint32_t *d_timingResultsTestCU = util::allocateGPUMemory(resultBufferLength);
    Barrier2* d_barrier = util::allocateGPUMemory<Barrier2>(1);

    util::hipCheck(hipMemset(d_barrier, 0, sizeof(Barrier2)));
    util::hipCheck(hipMemset(d_timingResultsBaseCU, 0, resultBufferLength * sizeof(uint32_t)));
    util::hipCheck(hipMemset(d_timingResultsTestCU, 0, resultBufferLength * sizeof(uint32_t)));

    std::vector<uint32_t> baseCUTimingResultsBuffer;
    std::vector<uint32_t> testCUTimingResultsBuffer;

    util::hipCheck(hipDeviceSynchronize());

    hipStream_t baseStream = util::createStreamForCU(baseCU);
    hipStream_t testStream = util::createStreamForCU(testCU);

    cuShareScalarL1LogicalKernel<<<1, 1, 0, baseStream>>>(d_pChaseArrayBaseCU, d_timingResultsBaseCU, d_barrier, steps, true);
    cuShareScalarL1LogicalKernel<<<1, 1, 0, testStream>>>(d_pChaseArrayTestCU, d_timingResultsTestCU, d_barrier, steps, false);

    util::hipCheck(hipDeviceSynchronize());
    
    baseCUTimingResultsBuffer = util::copyFromDevice(d_timingResultsBaseCU, resultBufferLength);
    testCUTimingResultsBuffer = util::copyFromDevice(d_timingResultsTestCU, resultBufferLength);

    baseCUTimingResultsBuffer.erase(baseCUTimingResultsBuffer.begin());
    testCUTimingResultsBuffer.erase(testCUTimingResultsBuffer.begin());

    util::hipCheck(hipStreamDestroy(baseStream));
    util::hipCheck(hipStreamDestroy(testStream));

    util::hipCheck(hipFree(d_pChaseArrayBaseCU));
    util::hipCheck(hipFree(d_pChaseArrayTestCU));
    util::hipCheck(hipFree(d_timingResultsBaseCU));
    util::hipCheck(hipFree(d_timingResultsTestCU));
    util::hipCheck(hipFree(d_barrier));

    return {{ baseCUTimingResultsBuffer, testCUTimingResultsBuffer }};
}

namespace benchmark {
    namespace amd {
        std::set<std::set<uint32_t>> measureCuShareScalarL1Logical(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes) {
            util::hipDeviceReset();

            DisjointSet dsu;

            std::set<uint32_t> skipLUT;

            for (uint32_t k = 0; k < util::getNumberOfComputeUnits(); ++k) 
            {
                std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>> testCUToTimingResults;

                for (uint32_t i = 0; i < util::getNumberOfComputeUnits(); ++i) 
                { 
                    if (k == i || skipLUT.contains(i)) continue;

                    auto timingsOpt = cuSharescalarL1LogicalLauncher(scalarL1SizeBytes, scalarL1FetchGranularityBytes, k, i);
                    if (!timingsOpt.has_value()) 
                    {
                        std::cout << "Skipping CU " << i << " because it did not return any results." << std::endl;
                        return {};
                    }

                    testCUToTimingResults[i] = timingsOpt.value();
                }

                auto mostLikelyPartners = util::detectShareChangePoint(testCUToTimingResults);
                
                if (!mostLikelyPartners.empty()) 
                {
                    for (auto partnerId : mostLikelyPartners) 
                    {
                        dsu.unite(k, partnerId);
                    }
                    skipLUT.insert(k);
                } 
                else 
                {
                    dsu.add(k);
                    skipLUT.insert(k);
                }
            }

            return dsu.getEquivalenceClasses();
        } 
    }
}

*/