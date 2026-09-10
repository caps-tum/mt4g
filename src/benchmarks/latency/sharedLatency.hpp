#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure the latency of shared memory accesses.
     *
     * @return Average latency in cycles.
     */
    CacheLatencyResult measureSharedMemoryLatency();
}