#pragma once

#include <cstddef>
#include <optional>

#include "utils/hip/memory.hpp"

namespace benchmark {
    /**
     * @brief Measure the latency of main memory accesses.
     *
     * @param allocType               Allocator used for the pointer-chase array.
     * @param cacheLineSizeBytes      Largest line size of all caches (L1, L2, L3), if all are known; the p-chase stride.
     * @param lastLevelCacheSizeBytes Size of the last level cache, sets the p-chase array size.
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureMainMemoryLatency(util::AllocatorType allocType = util::AllocatorType::HipMalloc,
                                                std::optional<size_t> cacheLineSizeBytes = std::nullopt,
                                                std::optional<size_t> lastLevelCacheSizeBytes = std::nullopt);
}