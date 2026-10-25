#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>

static constexpr auto SAMPLE_SIZE = 2048;// 2048 Loads should suffice to rule out random flukes
static constexpr auto WARMUP_HOPS = SAMPLE_SIZE;
static constexpr size_t SCRUB_STRIDE_BYTES = 64;

__global__ void mainMemoryLatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Continue from the warm-up position to avoid immediately repeating its path.
    for (uint32_t i = 0; i < WARMUP_HOPS; ++i) {
        index = __forceBypassAllCacheReads(pChaseArray, index);
    }

    // Do not load from the CU cache or L2.
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        uint64_t start = __timer();
        index = __forceBypassAllCacheReads(pChaseArray, index); 
        uint64_t end = __timer(); 

        s_timings[i] = end - start;
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }

    // Keep the final pointer value observable without overwriting a sample.
    timingResults[SAMPLE_SIZE] = index;
}

__global__ void mainMemoryLatencyInitKernel(uint32_t *destination, const uint32_t *source, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        destination[index] = source[index];
    }
}

__global__ void mainMemoryLatencyScrubKernel(const uint32_t *source, size_t count,
                                             uint32_t *sink) {
    constexpr size_t wordsPerCacheLine = SCRUB_STRIDE_BYTES / sizeof(uint32_t);
    const size_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t threadCount = gridDim.x * blockDim.x;
    uint32_t accumulator = 0;

    // Touch one word per assumed 64-byte cache line to create cache pressure.
    // This may evict measured-array data, but is not a guaranteed cache flush.
    for (size_t index = threadIndex * wordsPerCacheLine;
         index < count;
         index += threadCount * wordsPerCacheLine) {
        accumulator ^= source[index];
    }

    sink[threadIndex] = accumulator;
}

std::vector<uint32_t> mainMemoryLatencyLauncher(size_t arraySizeBytes, size_t strideBytes,
                                                util::AllocatorType allocType) {
    util::hipDeviceReset(); 

    std::vector<uint32_t> hostChaseArray = util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes);

    uint32_t *d_pChaseArray = util::allocateMemory<uint32_t>(hostChaseArray.size(), allocType);

    uint32_t *d_initArray = util::allocateGPUMemory(hostChaseArray);

    uint32_t threads = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t blocks = util::getNumberOfComputeUnits() * util::maxBlocksPerMultiProcessor();

    // Use the same GPU first-touch initialization path for every allocator.
    mainMemoryLatencyInitKernel<<<blocks, threads>>>(d_pChaseArray, d_initArray, hostChaseArray.size());
    util::hipCheck(hipDeviceSynchronize());

    // Apply cache pressure before measuring the pointer chase.
    const size_t scrubSinkSize = static_cast<size_t>(blocks) * threads;
    uint32_t *d_scrubSink = util::allocateGPUMemory(scrubSinkSize);
    mainMemoryLatencyScrubKernel<<<blocks, threads>>>(
        d_initArray, hostChaseArray.size(), d_scrubSink);
    util::hipCheck(hipDeviceSynchronize());

    util::hipCheck(hipFree(d_scrubSink));
    util::hipCheck(hipFree(d_initArray));
    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE + 1);

    util::hipCheck(hipDeviceSynchronize());
    mainMemoryLatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    util::freeMemory(d_pChaseArray, allocType);
    util::hipCheck(hipFree(d_timingResults));
    return timingResultBuffer;
}

namespace benchmark {
    CacheLatencyResult measureMainMemoryLatency(util::AllocatorType allocType) {

        auto timings = mainMemoryLatencyLauncher(1 * GiB, 1 * KiB, allocType);

        CacheLatencyResult result {
            timings,
            util::average(timings),
            util::percentile(timings, 0.5),
            util::percentile(timings, 0.95),
            util::stdev(timings),
            timings.size(),
            SAMPLE_SIZE,
            CYCLE,
            PCHASE
        };

        return result;
    }
}