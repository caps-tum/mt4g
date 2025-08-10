#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure the latency of main memory accesses.
     *
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureMainMemoryLatency();
}