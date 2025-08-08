#pragma once
#include <cstddef>
#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the penalty incurred on a scalar L1 cache miss.
         *
         * The cache is evicted before timing loads and subtracting the
         * measured scalar L1 hit latency.
         *
         * @param scalarL1CacheSizeBytes       Size of the scalar L1 cache in bytes.
         * @param scalarL1FetchGranularityBytes Line or sector size used for the test.
         * @param scalarL1Latency              Previously measured scalar L1 hit latency.
         * @return Additional latency caused by a scalar L1 miss.
         */
        double measureScalarL1MissPenalty(size_t scalarL1CacheSizeBytes,
                                          size_t scalarL1FetchGranularityBytes,
                                          double scalarL1Latency);
    }
}
