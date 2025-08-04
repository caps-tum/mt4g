#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the latency of the L3 cache on AMD GPUs.
         * @param l2FetchGranularityBytes The fetch granularity of the L2 cache in bytes.
         * 
         * @return Average latency in cycles.
         */
        CacheLatencyResult measureL3Latency(size_t l2SizeBytes, size_t l2FetchGranularityBytes);
    }
}