#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure latency of read-only data cache accesses.
         *
         * @return Average latency in cycles.
         */
        CacheLatencyResult measureReadOnlyLatency();
    }
}