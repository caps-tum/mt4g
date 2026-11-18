#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <iostream>

static constexpr auto SAMPLE_SIZE = 2048;// 2048 Loads should suffice to rule out random flukes
static constexpr auto WARMUP_HOPS = SAMPLE_SIZE;
static constexpr auto MIN_ARRAY_SIZE = 512 * MiB;
static constexpr auto LLC_SIZE_FACTOR = 4;// array size relative to the last level cache
static constexpr auto DEFAULT_STRIDE = 256;// used if no cache line size is reported

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

// Loads (and thereby caches) the buffer, evicting older data from all cache levels
__global__ void evictCachesKernel(const uint32_t *buffer, size_t count, uint32_t *sink) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    uint32_t accumulator = 0;
    for (size_t i = index; i < count; i += stride) {
        accumulator ^= buffer[i];
    }
    sink[index] = accumulator;
}

// Largest line size of the regular caches (L1, L2, L3), so every hop loads a new line in all of them
static size_t chaseStrideBytes(std::optional<size_t> l1LineSizeBytes) {
    if (!l1LineSizeBytes.has_value()) {
        l1LineSizeBytes = util::getL1LineSizeBytes();
    }
    size_t stride = 0;
    for (auto lineSize : {l1LineSizeBytes, util::getL2LineSizeBytes(), util::getL3LineSizeBytes()}) {
        stride = util::max(stride, lineSize.value_or(0));
    }
    if (stride == 0) {
        std::cout << "[Main Memory] No cache line size available, using the default p-chase stride of " << DEFAULT_STRIDE << " bytes" << std::endl;
        return DEFAULT_STRIDE;
    }
    return stride;
}

static size_t chaseArraySizeBytes() {
    size_t lastLevelCacheBytes = util::max(util::getL3SizeBytes().value_or(0),
                                           static_cast<size_t>(util::getDeviceProperties().l2CacheSize));
    return util::max(MIN_ARRAY_SIZE, LLC_SIZE_FACTOR * lastLevelCacheBytes);
}

std::vector<uint32_t> mainMemoryLatencyLauncher(size_t arraySizeBytes, size_t strideBytes,
                                                util::AllocatorType allocType) {
    util::hipDeviceReset(); 

    std::vector<uint32_t> hostChaseArray = util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes);

    uint32_t *d_pChaseArray = util::allocateMemory<uint32_t>(hostChaseArray.size(), allocType);

    uint32_t *d_initArray = util::allocateGPUMemory(hostChaseArray);

    uint32_t threads = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t blocks = util::getNumberOfComputeUnits() * util::getMaxBlocksPerMultiProcessor();

    // Use the same GPU first-touch initialization path for every allocator.
    mainMemoryLatencyInitKernel<<<blocks, threads>>>(d_pChaseArray, d_initArray, hostChaseArray.size());
    util::hipCheck(hipDeviceSynchronize());

    // The initialization left parts of the p-chase array in the caches. d_initArray has the
    // same size and is no longer needed, so loading it evicts them.
    uint32_t *d_evictionSink = util::allocateGPUMemory(static_cast<size_t>(blocks) * threads);
    evictCachesKernel<<<blocks, threads>>>(d_initArray, hostChaseArray.size(), d_evictionSink);
    util::hipCheck(hipDeviceSynchronize());

    util::hipCheck(hipFree(d_evictionSink));
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
    CacheLatencyResult measureMainMemoryLatency(util::AllocatorType allocType, std::optional<size_t> l1LineSizeBytes) {

        auto timings = mainMemoryLatencyLauncher(chaseArraySizeBytes(), chaseStrideBytes(l1LineSizeBytes), allocType);

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