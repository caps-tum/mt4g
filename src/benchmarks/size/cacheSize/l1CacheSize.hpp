#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    /**
     * @brief Measure the effective size of the L1 cache.
     *
     * @param cacheFetchGranularityBytes Line or sector size in bytes.
     * @return Detected cache size.
     */
    CacheSizeResult measureL1Size(size_t cacheFetchGranularityBytes);
}