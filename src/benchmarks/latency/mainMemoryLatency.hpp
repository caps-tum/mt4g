#pragma once

#include <cstddef>
#include <optional>

#include "utils/hip/memory.hpp"

namespace benchmark {
    /**
     * @brief Measure the latency of main memory accesses.
     *
     * @param allocType       Allocator used for the pointer-chase array.
     * @param l1LineSizeBytes L1 line size measured by mt4g, otherwise the reported one is used.
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureMainMemoryLatency(util::AllocatorType allocType = util::AllocatorType::HipMalloc,
                                                std::optional<size_t> l1LineSizeBytes = std::nullopt);
}