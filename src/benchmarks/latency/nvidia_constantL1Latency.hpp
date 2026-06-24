#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure latency of the constant memory L1 cache.
         *
         * @return Average latency in cycles.
         */
        CacheLatencyResult measureConstantL1Latency();
    }
}