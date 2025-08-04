#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Check if read-only and regular L1 caches share the same physical bank.
         *
         * @param readOnlyCacheSizeBytes      Size of the read-only cache.
         * @param readOnlyFetchGranularityBytes Line/sector size of the read-only cache.
         * @param readOnlyLatency             Latency of read-only cache hits.
         * @param readOnlyMissPenalty         Miss penalty of the read-only cache.
         * @param l1CacheSizeBytes            Size of the regular L1 cache.
         * @param l1FetchGranularityBytes     Line/sector size of the regular L1.
         * @param l1Latency                   Latency of regular L1 hits.
         * @param l1MissPenalty               Miss penalty of the regular L1.
         * @return True if both caches share the same bank, otherwise false.
         */
        bool measureReadOnlyAndL1Shared(size_t readOnlyCacheSizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyLatency, double readOnlyMissPenalty, size_t l1CacheSizeBytes,size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty);
    }
}