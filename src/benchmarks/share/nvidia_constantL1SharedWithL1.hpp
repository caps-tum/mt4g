#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Check if constant L1 and regular L1 share the same physical bank.
         *
         * @param constantL1CacheSizeBytes      Size of constant L1 cache.
         * @param constantL1FetchGranularityBytes Line/sector size of constant cache.
         * @param constantL1Latency             Latency of constant L1 hits.
         * @param constantL1MissPenalty         Miss penalty of constant L1.
         * @param l1CacheSizeBytes              Size of regular L1.
         * @param l1FetchGranularityBytes       Line/sector size of regular L1.
         * @param l1Latency                     Latency of regular L1 hits.
         * @param l1MissPenalty                 Miss penalty of regular L1.
         * @return True if both caches share the same bank, otherwise false.
         */
        bool measureConstantL1AndL1Shared(size_t constantL1CacheSizeBytes, size_t constantL1FetchGranularityBytes, double constantL1Latency, double constantL1MissPenalty, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty);
    }
}