#pragma once

#include <cstddef>
#include <optional>

#include "utils/hip/memory.hpp"

namespace benchmark {
    /**
     * @brief Measure the latency of main memory accesses.
     *
     * @param allocType       Allocator used for the pointer-chase array.
     * @param l2LineSizeBytes L2 cache line size reported/measured by mt4g. It sets the
     *                        stride of the cache scrub that runs before the measurement.
     *                        If unknown, the scrub touches every 32-bit word instead.
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureMainMemoryLatency(
        util::AllocatorType allocType = util::AllocatorType::HipMalloc,
        std::optional<size_t> l2LineSizeBytes = std::nullopt);
}