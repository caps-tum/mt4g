#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <numeric>

static constexpr auto WARMUP_REPS = 128;

static constexpr auto MIN_REPS = 2;
static constexpr auto MAX_REPS = 262144; // 2 ^ 18

static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void sL1WriteBandwidthKernel(uint32v4* __restrict__ dst, uint64_t* __restrict__ timings, uint32_t waveSize, size_t elementsPerWave, size_t reps) 
{
    uint32_t tid = threadIdx.x;
    uint32_t waveid = tid / waveSize;

    uint32_t scalarWaveid;
    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "v_readfirstlane_b32 %0, %1" 
        : "=s"(scalarWaveid) 
        : "v"(waveid)
    );
    #endif

    const uint32_t base = scalarWaveid * elementsPerWave;
    uint32v4* addr = dst + base;

    uint32v4 dummy {scalarWaveid, scalarWaveid + 1, scalarWaveid + 2, scalarWaveid + 3};

    __syncthreads();

    // warm up
    for (uint32_t rep = 0; rep < WARMUP_REPS; ++rep)
    {
        uint32v4* offset = addr;

        for (size_t i = 0; i < elementsPerWave; i += 4)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "s_store_dwordx4 %0, %1, 0\n\t"
                "s_store_dwordx4 %0, %1, 16\n\t"
                "s_store_dwordx4 %0, %1, 32\n\t"
                "s_store_dwordx4 %0, %1, 48\n\t"
                :
                : "s"(dummy), "s"(offset)
                : "memory"
            );
            #endif

            offset += 4;
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        :
        : "memory"
    );
    #endif

    __syncthreads();

    if (tid == 0) {
        uint64_t start;
        
        #ifdef __HIP_PLATFORM_AMD__
        __asm__ volatile (
            "s_memtime %0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(start) 
            :
            : "memory"
        );
        #endif

        timings[0] = start;
    }

    __syncthreads();

    for (uint32_t rep = 0; rep < reps; ++rep) 
    {
        uint32v4* offset = addr;

        for (size_t i = 0; i < elementsPerWave; i += 4)
        {
            #ifdef __HIP_PLATFORM_AMD__
            __asm__ volatile (
                "s_store_dwordx4 %0, %1, 0\n\t"
                "s_store_dwordx4 %0, %1, 16\n\t"
                "s_store_dwordx4 %0, %1, 32\n\t"
                "s_store_dwordx4 %0, %1, 48\n\t"
                :
                : "s"(dummy), "s"(offset)
                : "memory"
            );
            #endif

            offset += 4;
        }
    }

    #ifdef __HIP_PLATFORM_AMD__
    __asm__ volatile (
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        :
        : "memory"
    );
    #endif

    __syncthreads();

    if (tid == 0) 
    {
        uint64_t end;

        #ifdef __HIP_PLATFORM_AMD__
        __asm__ volatile (
            "s_memtime %0\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            : "=s"(end) 
            :
            : "memory"
        );
        #endif

        timings[1] = end;
    }
}

static std::tuple<uint64_t, double, double> sL1WriteBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps) 
{
    size_t numElements = arraySizeBytes / sizeof(uint32v4);
    uint32_t waveSize = util::getDeviceProperties().warpSize;
    uint32_t numWaves = numThreads / waveSize;
    size_t elementsPerWave = numElements / numWaves;
    
    uint32v4* d_dstArr = util::allocateGPUMemory<uint32v4>(numElements);
    uint64_t* d_timings = util::allocateGPUMemory<uint64_t>(2);
    
    sL1WriteBandwidthKernel<<<1, numThreads>>>(d_dstArr, d_timings, waveSize, elementsPerWave, reps);

    std::vector<uint64_t> t = util::copyFromDevice<uint64_t>(d_timings, 2);

    uint64_t cycles = t[1] - t[0];
    double gpuClockHz = util::getDeviceProperties().clockRate * 1000.0;
    double timeS = (double)cycles / gpuClockHz;
    double dataGiB = (double)arraySizeBytes * reps / (1 * GiB);

    return {cycles, timeS, dataGiB / timeS};
}

namespace benchmark {
    namespace amd {
        double measureScalarL1WriteBandwidth(size_t arraySizeBytes)
        {
            std::vector<double> results(ROUNDS);

            uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;
            size_t maxReps = MAX_REPS;

            for (uint32_t i = 0; i < ROUNDS; ++i) 
            {
                results[i] = std::get<2>(sL1WriteBandwidthLauncher(arraySizeBytes, maxNumThreads, maxReps));
            }

            return util::average(results);
        }

        CacheBandwidthResult measureScalarL1WriteBandwidthSweep(size_t arraySizeBytes) 
        {            
            uint32_t minNumThreads = util::getDeviceProperties().warpSize;
            uint32_t maxNumThreads = util::getDeviceProperties().maxThreadsPerBlock;
            size_t minReps = MIN_REPS;
            size_t maxReps = MAX_REPS;

            CacheBandwidthResult result{};
            result.measuredBandwidth = 0.0;
            result.dataBytes = arraySizeBytes;
            result.cycles = 0;
            result.time = 0.0;
            result.numThreads = 0;
            result.numBlocks = 1;
            result.numReps = 0;

            for (uint32_t numThreads = minNumThreads; numThreads <= maxNumThreads; numThreads *= 2)
            {
                std::vector<double> bandwidthResults;

                result.threadsTested.push_back(numThreads);

                for (size_t reps = minReps; reps <= maxReps; reps *= 2)
                {
                    if (numThreads == minNumThreads)
                    {
                        result.repsTested.push_back(reps);
                    }

                    auto [cycles, timeS, bandwidth] = sL1WriteBandwidthLauncher(arraySizeBytes, numThreads, reps);
                    
                    bandwidthResults.push_back(bandwidth);

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.cycles = cycles;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numReps = reps;
                    }
                }

                result.bandwidthGridGiBs.push_back(bandwidthResults);
            }

            return result;
        }
    }
}