#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <iostream>
#include <stdexcept>

static constexpr auto SAMPLE_SIZE = 2048;// 2048 Loads should suffice to rule out random flukes
static constexpr auto WARMUP_HOPS = SAMPLE_SIZE;
static constexpr auto MIN_ARRAY_SIZE = 512 * MiB;
static constexpr auto LLC_SIZE_FACTOR = 4;// array size relative to the last level cache
static constexpr auto DEFAULT_STRIDE = 256;// used if no cache line size is reported
// The chain stays in a window of this size before moving on: few pages for the TLB, but still many DRAM rows per bank
static constexpr auto CHASE_WINDOW_SIZE = 8 * MiB;

__global__ void mainMemoryLatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Warm-up: warms the TLB, the measured hops then load other lines of the same pages
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

    // final index, checked on the host
    timingResults[SAMPLE_SIZE] = index;
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

__global__ void mainMemoryLatencyInitKernel(uint32_t *destination, const uint32_t *source, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        destination[index] = source[index];
    }
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

// Order in which the chain visits the windows, starting with the window of element 0
static std::vector<size_t> windowVisitOrder(const std::vector<uint32_t> &chaseArray, size_t windowWords) {
    std::vector<size_t> order;
    uint32_t index = 0;
    do {
        const size_t window = index / windowWords;
        if (order.empty() || order.back() != window) {
            order.push_back(window);
        }
        index = chaseArray[index];
    } while (index != 0);
    return order;
}

// The node the chase must end on after the given number of hops from node 0
static uint32_t expectedFinalIndex(const std::vector<uint32_t> &chaseArray, size_t hops) {
    uint32_t index = 0;
    for (size_t i = 0; i < hops; ++i) {
        index = chaseArray[index];
    }
    return index;
}

static size_t chaseArraySizeBytes() {
    size_t lastLevelCacheBytes = util::max(util::getL3SizeBytes().value_or(0),
                                           static_cast<size_t>(util::getDeviceProperties().l2CacheSize));
    return util::max(MIN_ARRAY_SIZE, LLC_SIZE_FACTOR * lastLevelCacheBytes);
}

std::vector<uint32_t> mainMemoryLatencyLauncher(size_t arraySizeBytes, size_t strideBytes,
                                                util::AllocatorType allocType) {
    util::hipDeviceReset(); 

    std::vector<uint32_t> hostChaseArray = util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes, CHASE_WINDOW_SIZE);
    const size_t windowWords = util::max(CHASE_WINDOW_SIZE / strideBytes, static_cast<size_t>(1)) * strideBytes / sizeof(uint32_t);
    const std::vector<size_t> windowOrder = windowVisitOrder(hostChaseArray, windowWords);
    if (windowOrder.size() < 2) {
        throw std::logic_error("Main memory latency: the p-chase array must be larger than one window");
    }

    uint32_t *d_pChaseArray = util::allocateMemory<uint32_t>(hostChaseArray.size(), allocType);

    uint32_t *d_initArray = util::allocateGPUMemory(hostChaseArray);

    uint32_t threads = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t blocks = util::getNumberOfComputeUnits() * util::getMaxBlocksPerMultiProcessor();

    // Use the same GPU first-touch initialization path for every allocator.
    mainMemoryLatencyInitKernel<<<blocks, threads>>>(d_pChaseArray, d_initArray, hostChaseArray.size());
    util::hipCheck(hipDeviceSynchronize());

    util::hipCheck(hipFree(d_initArray));

    // Evict: load the windows in the chain's visiting order, one kernel per window (kernels in a stream run in order).
    // The windows of the measured hops are loaded first, so they are the oldest data in all caches.
    uint32_t *d_evictionSink = util::allocateGPUMemory(static_cast<size_t>(blocks) * threads);
    for (size_t window : windowOrder) {
        const size_t firstWord = window * windowWords;
        evictCachesKernel<<<blocks, threads>>>(d_pChaseArray + firstWord, util::min(windowWords, hostChaseArray.size() - firstWord), d_evictionSink);
    }
    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipFree(d_evictionSink));

    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE + 1);

    util::hipCheck(hipDeviceSynchronize());
    mainMemoryLatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE + 1);

    // A wrong final index means the GPU did not see the generated chain (e.g. a broken allocation)
    const uint32_t finalIndex = timingResultBuffer.back();
    timingResultBuffer.pop_back();
    if (finalIndex != expectedFinalIndex(hostChaseArray, WARMUP_HOPS + SAMPLE_SIZE)) {
        throw std::runtime_error("Main memory latency: the GPU did not follow the p-chase chain, the measurement is invalid");
    }

    util::freeMemory(d_pChaseArray, allocType);
    util::hipCheck(hipFree(d_timingResults));
    return timingResultBuffer;
}

namespace benchmark {
    // TODO: report the array size, stride and window size in the result
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