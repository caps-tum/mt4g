#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure latency of scalar L1 cache accesses.
         *
         * @return Average latency in cycles.
         */
        CacheLatencyResult measureScalarL1Latency();
    }
}