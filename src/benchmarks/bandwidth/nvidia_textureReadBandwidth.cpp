#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <tuple>
#include <vector>

static constexpr auto WARMUP_REPS = 128;
static constexpr auto MS_PER_SECOND = 1000.0; // ms

static constexpr size_t LOADS_PER_GROUP = 8;
static constexpr size_t GROUP_LOAD_STRIDE = 0;

__global__ void textureReadBandwidthKernel([[maybe_unused]] hipTextureObject_t tex, uint4* dst, size_t totalElements, size_t reps, [[maybe_unused]] size_t groupLoadStride)
{
    uint4 dummy {0, 0, 0, 0};

    #ifdef __HIP_PLATFORM_NVIDIA__
    for (size_t rep = 0; rep < reps; rep += LOADS_PER_GROUP)
    {
        for (size_t i = threadIdx.x; i < totalElements; i += blockDim.x)
        {
            #pragma unroll
            for (size_t k = 0; k < LOADS_PER_GROUP; ++k)
            {
                const int4 loaded = tex1Dfetch<int4>(tex, static_cast<int>(i + k * groupLoadStride));

                dummy.x ^= loaded.x;
                dummy.y ^= loaded.y;
                dummy.z ^= loaded.z;
                dummy.w ^= loaded.w;
            }
        }
    }
    #endif

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


static std::tuple<double, double> textureReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, size_t reps, hipStream_t stream)
{
    const size_t totalElements = arraySizeBytes / sizeof(int4);

    int4 *d_srcArr = util::allocateGPUMemory<int4>(totalElements);
    uint4 *d_dstArr = util::allocateGPUMemory<uint4>(numThreads);

    hipTextureObject_t tex = util::createTextureObject<int4>(d_srcArr, totalElements);

    // Warm up
    textureReadBandwidthKernel<<<1, numThreads, 0, stream>>>(tex, d_dstArr, totalElements, WARMUP_REPS, GROUP_LOAD_STRIDE);

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start, stream));
    textureReadBandwidthKernel<<<1, numThreads, 0, stream>>>(tex, d_dstArr, totalElements, reps, GROUP_LOAD_STRIDE);
    util::hipCheck(hipEventRecord(end, stream));
    util::hipCheck(hipDeviceSynchronize());

    const double elapsedMs = util::getElapsedTimeMs(start, end);

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));
    util::hipCheck(hipDestroyTextureObject(tex));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));

    const double timeS = elapsedMs / MS_PER_SECOND;
    const double dataGiB = (double) (totalElements * sizeof(int4)) * reps / (1 * GiB);

    return {timeS, dataGiB / timeS};
}


namespace benchmark
{
    namespace nvidia
    {
        CacheBandwidthResult measureTextureReadBandwidthSweep(size_t arraySizeBytes)
        {
            auto stream = util::createStreamForCU(0);

            uint32_t minNumThreads = util::getWarpSize();
            uint32_t maxNumThreads = util::getMaxThreadsPerBlock();
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

                    auto [timeS, bandwidth] = textureReadBandwidthLauncher(arraySizeBytes, numThreads, reps, stream);

                    bandwidthResults.push_back(bandwidth);

                    if (bandwidth > result.measuredBandwidth)
                    {
                        result.measuredBandwidth = bandwidth;
                        result.time = timeS;
                        result.numThreads = numThreads;
                        result.numReps = reps;
                    }
                }

                result.bandwidthGridGiBs.push_back(bandwidthResults);
            }

            util::hipCheck(hipStreamDestroy(stream));

            return result;
        }
    }
}
