#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure the read-only cache miss penalty.
         *
         * @param readonlyCacheSizeBytes      Size of the read-only cache.
         * @param readonlyCacheLineSizeBytes  Line/sector size of the cache.
         * @param readOnlyLatency             Latency of read-only cache hits.
         * @return Miss penalty in cycles.
         */
        double measureReadOnlyMissPenalty(size_t readonlyCacheSizeBytes, size_t readonlyCacheLineSizeBytes, double readOnlyLatency);
   }
}