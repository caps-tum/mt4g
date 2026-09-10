#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Determine how many scalar L1 caches are shared between cores.
         *
         * @param scalarL1SizeBytes          Size of the scalar L1 cache.
         * @param scalarL1FetchGranularityBytes fetch granularity for the cache.
         * @return Number of cores per scalar L1 cache.
         */
        uint32_t measureScalarL1Amount(size_t scalarL1SizeBytes, size_t scalarL1FetchGranularityBytes);
    }
}