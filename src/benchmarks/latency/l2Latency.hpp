#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure the latency of L2 cache hits.
     *
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureL2Latency();
}