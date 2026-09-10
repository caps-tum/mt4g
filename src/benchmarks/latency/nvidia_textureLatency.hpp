#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure latency of texture cache accesses.
         *
         * @return Average latency in cycles.
         */
        CacheLatencyResult measureTextureLatency();
    }
}