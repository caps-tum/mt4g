#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure the constant L1 miss penalty.
         *
         * @param constantL1CacheSizeBytes      Size of the constant L1 cache.
         * @param constantL1CacheLineSizeBytes Line/sector size of the constant cache.
         * @param constantL1Latency             Latency of constant L1 hits.
         * @return Miss penalty in cycles.
         */
        double measureConstantL1MissPenalty(size_t constantL1CacheSizeBytes, size_t constantL1CacheLineSizeBytes, double constantL1Latency);
    }
}