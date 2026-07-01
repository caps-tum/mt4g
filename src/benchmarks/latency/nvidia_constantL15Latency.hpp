#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure latency of the constant L1.5 cache using a given stride.
         *
         * @param constantL1SizeBytes Size of the constant cache region.
         * @param stride              Access stride in elements.
         * @return Average latency in cycles.
         */
        CacheLatencyResult measureConstantL15Latency(size_t constantL1SizeBytes, size_t stride);
    }
}