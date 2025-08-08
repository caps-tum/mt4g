#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    /**
     * @brief Measure the L1 miss penalty using shared data.
     *
     * @param l1CacheSizeBytes      Cache size of each L1 in bytes.
     * @param l1FetchGranularityBytes Cache line or sector size in bytes.
     * @param l1Latency             Measured L1 latency.
     * @return Miss penalty in cycles.
     */
    double measureL1MissPenalty(size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency);
}