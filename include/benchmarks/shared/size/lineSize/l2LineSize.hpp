#pragma once

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    /**
     * @brief Determine the line size of the L2 cache.
     *
     * @param cacheSizeBytes          Total cache size under test.
     * @param cacheFetchGranularityBytes Sector size used during the measurement.
     * @return Detected line size in bytes if successful.
     */
    CacheSizeResult measureL2LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes);
}