#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>

__global__ void l1ReadBandwidthCUKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, uint64_t* __restrict__ timing_result, size_t elementsPerThread, size_t reps) 
{
    const uint32_t tid = threadIdx.x;
    const uint32v4* base = src + tid * elementsPerThread;

    const uint64_t addr0 = reinterpret_cast<uint64_t>(base);
    uint64_t start, end; // timer

    uint32v4 dummy {0, 0, 0, 0};

    // Warm up L1
    for (size_t rep = 0; rep < 32; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            uint32v4 loaded;
            asm volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr0 + i * sizeof(uint32v4))
                : "memory"
            );

            dummy.x ^= loaded.x;
        }
    }
    
    __asm__ volatile (
        "s_waitcnt vmcnt(0)\n\t"
        :
        :
        : "memory"
    );

    __syncthreads();

    if (tid == 0)
    {
        __asm__ volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_memtime %0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(start)
            :
            : "memory"
        );
    }

    __syncthreads();

    for (size_t rep = 0; rep < reps; ++rep)
    {
        for (size_t i = 0; i < elementsPerThread; ++i)
        {
            uint32v4 loaded;
            __asm__ volatile (
                "flat_load_dwordx4 %0, %1\n\t"
                : "=v"(loaded)
                : "v"(addr0 + i * sizeof(uint32v4))
                : "memory"
            );

            dummy.x ^= loaded.x;
        }
    }

    __asm__ volatile (
        "s_waitcnt vmcnt(0)\n\t"
        :
        :
        : "memory"
    );    

    __syncthreads();

    if (tid == 0)
    {
        __asm__ volatile (
            "s_waitcnt lgkmcnt(0)\n\t"
            "s_memtime %0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(end)
            :
            : "memory"
        );

        *timing_result = end - start;
    }

    dst[tid] = dummy; // prevent dead code elimination
}

static double l1ReadBandwidthCULauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps) 
{
    size_t totalElements = arraySizeBytes / sizeof(uint32v4);
    size_t elementsPerThread = totalElements / numThreads;

    // std::cout << "number of threads : " << numThreads << std::endl;
    // std::cout << "total elements : " << totalElements << std::endl;
    // std::cout << "elementsPerThread : " << elementsPerThread << std::endl;
    // std::cout << "REPS : " << reps << std::endl;

    // Allocate device arrays           
    uint32v4 *d_srcArr = util::allocateGPUMemory<uint32v4>(totalElements);
    uint32v4 *d_dstArr = util::allocateGPUMemory<uint32v4>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    // std::cout << "running the kernel" << std::endl;

    // Run the kernel
    l1ReadBandwidthCUKernel<<<1, numThreads>>>(d_dstArr, d_srcArr, d_timingResult, elementsPerThread, reps);

    // std::cout << "trying to get timings" << std::endl;

    // Get the timings from the device
    std::vector<uint64_t> timingResult = util::copyFromDevice<uint64_t>(d_timingResult, 1);

    // std::cout << "got the cycles : " << timingResult[0] << std::endl;

    // calculate the bandwidth
    double gpuClockHz = util::getDeviceProperties().clockRate * 1000;
    double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);
    double timeS = (double)timingResult[0] / gpuClockHz;

    // std::cout << "dataGiB : " << dataGiB << std::endl;
    // std::cout << "time s : " << timeS << std::endl;
    // std::cout << "BW : " << dataGiB / timeS << std::endl;
    
    return dataGiB / timeS;
}


namespace benchmark 
{
    double measureL1ReadBandwidthCU(size_t arraySizeBytes) 
    {
        std::cout << "tested size: " << arraySizeBytes << std::endl;

        uint32_t minNumThreads = util::getDeviceProperties().warpSize;
        uint32_t maxNumThreads = util::min(util::getDeviceProperties().maxThreadsPerBlock, 1024);
        size_t minReps = 2;
        size_t maxReps = 262144;

        std::vector<std::vector<double>> results;

        for (uint32_t numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads *= 2)
        {
            std::vector<double> repsResults = {
                l1ReadBandwidthCULauncher(arraySizeBytes, numThreads, minReps),
                l1ReadBandwidthCULauncher(arraySizeBytes, numThreads, minReps * 2)
            };

            for (size_t reps = minReps * 4; reps <= maxReps; reps *= 2)
            {
                repsResults.push_back(l1ReadBandwidthCULauncher(arraySizeBytes, numThreads, reps));
            }

            results.push_back(repsResults);
        }

        double max = -1.0;

        for (size_t i = 0; i < results.size(); ++i)
        {
            //std::cout << "[ ";
            for (size_t j = 0; j < results[i].size(); ++j)
            {
                if (max < results[i][j])
                {
                    max = results[i][j];
                }

                // std::cout << results[i][j] << ", ";
            }
            //std::cout << " ]" << std::endl;
        }

        return max;
    }
}
