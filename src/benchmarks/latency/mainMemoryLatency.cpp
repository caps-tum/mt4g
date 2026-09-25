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

    // Warm up the execution path, then continue from the resulting index. Do
    // not reset to zero: the timed loads must visit nodes that were not read by
    // this pointer chase during warm-up.
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

    // Keep the dependent pointer chase observable without modifying a timing
    // sample. The launcher allocates one extra element for this sink.
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

    // Across all threads, read one uint32_t every 64 bytes from the source.
    // Assuming 64-byte cache lines, this touches every source cache line;
    // a cache miss can fetch the entire line.
    // These ordinary loads fill caches with source-buffer lines, aiming to
    // evict measured-array lines left by initialization.
    // This is not a guaranteed cache flush.
    for (size_t index = threadIndex * wordsPerCacheLine;
         index < count;
         index += threadCount * wordsPerCacheLine) {
        accumulator ^= source[index];
    }

    // Store each thread's accumulator separately so the loads have an
    // observable result and threads do not overwrite each other's output.
    sink[threadIndex] = accumulator;
}

std::vector<uint32_t> mainMemoryLatencyLauncher(size_t arraySizeBytes, size_t strideBytes,
                                                util::AllocatorType allocType) {
    util::hipDeviceReset(); 

    // Initialize device Arrays
    // The vector contains: 1 GiB / 4 bytes = 268,435,456 uint32_t elements
    // The stride in elements is: 1 KiB / 4 bytes = 256 elements
    // There are: 268,435,456 / 256 = 1,048,576 chain nodes
    std::vector<uint32_t> hostChaseArray = util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes);

    // Allocate the measured array: this creates the 1 GiB allocation whose latency we want to measure
    uint32_t *d_pChaseArray = util::allocateMemory<uint32_t>(hostChaseArray.size(), allocType);

    // Allocate the temporary source
    uint32_t *d_initArray = util::allocateGPUMemory(hostChaseArray);

    uint32_t threads = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t blocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    // Use the same GPU first-touch initialization path for every allocator.
    mainMemoryLatencyInitKernel<<<blocks, threads>>>(d_pChaseArray, d_initArray, hostChaseArray.size());
    util::hipCheck(hipDeviceSynchronize());

    // Reuse the 1 GiB initialization source as a cache scrubber. Its active
    // cache-line footprint is four times the 256 MiB MALL on MI300A.
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