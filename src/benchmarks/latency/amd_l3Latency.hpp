#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"
#include "utils/hip/memory.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the latency of the L3 cache on AMD GPUs.
         *
         * @param l2SizeBytes            Total L2 cache size in bytes used to size the working set.
         * @param l2FetchGranularityBytes Fetch granularity of the L2 cache in bytes.
         * @param allocType               Allocator used for the pointer-chase working set.
         * @return Average L3 hit latency in cycles.
         */
        CacheLatencyResult measureL3Latency(
            size_t l2SizeBytes, size_t l2FetchGranularityBytes,
            util::AllocatorType allocType = util::AllocatorType::HipMalloc);
    }
}