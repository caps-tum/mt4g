#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Determine how many cores share the read-only data cache.
         *
         * @param readOnlySizeBytes          Cache size in bytes.
         * @param readOnlyFetchGranularityBytes fetch granularity.
         * @param readOnlyMissPenalty        Penalty for a cache miss.
         * @return Number of cores per cache.
         */
        std::optional<uint32_t> measureReadOnlyAmount(size_t readOnlySizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyMissPenalty);
    }
}