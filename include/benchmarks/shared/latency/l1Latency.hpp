#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure the load latency of L1 cache hits.
     *
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureL1Latency();
}