#pragma once
#include <cstddef>
#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the penalty incurred on an L3 cache miss.
         *
         * Pointer-chasing evicts the L3 cache before timing the accesses and
         * comparing them against the provided L3 hit latency.
         *
         * @param l3CacheSizeBytes       Size of the L3 cache in bytes.
         * @param l3FetchGranularityBytes Line or sector size used for the test.
         * @param l3Latency              Previously measured L3 hit latency.
         * @return Additional latency caused by an L3 miss.
         */
        double measureL3MissPenalty(size_t l3CacheSizeBytes,
                                    size_t l3FetchGranularityBytes,
                                    double l3Latency);
    }
}
