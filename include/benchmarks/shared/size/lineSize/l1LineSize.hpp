#pragma once

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    /**
     * @brief Determine the line size of the L1 cache.
     *
     * @param cacheSizeBytes          Total cache size under test.
     * @param cacheFetchGranularityBytes Sector size used during the measurement.
     * @return Detected line size in bytes if successful.
     */
    CacheSizeResult measureL1LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes);
}