#pragma once

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Determine the line size of the scalar L1 cache.
         *
         * @param cacheSizeBytes          Cache size used for measurement.
         * @param cacheFetchGranularityBytes Sector size used for measurement.
         * @return Detected line size in bytes if successful.
         */
        CacheSizeResult measureScalarL1LineSize(size_t cacheSizeBytes, size_t cacheFetchGranularityBytes);
    }
}