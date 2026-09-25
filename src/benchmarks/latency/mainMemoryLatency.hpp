#pragma once

#include <cstddef>

#include "utils/hip/memory.hpp"

namespace benchmark {
    /**
     * @brief Measure the latency of main memory accesses.
     *
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureMainMemoryLatency(
        util::AllocatorType allocType = util::AllocatorType::HipMalloc);
}