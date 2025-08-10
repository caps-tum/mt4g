#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the effective size of the L3 cache.
         *
         * @param cacheFetchGranularityBytes Cache line or sector size to use.
         * @return Detected cache size in bytes.
         */
        CacheSizeResult measureL3Size(size_t cacheFetchGranularityBytes);
    }
}