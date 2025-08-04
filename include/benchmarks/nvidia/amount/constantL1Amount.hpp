#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Determine how many cores share the constant L1 cache.
         *
         * @param constantL1SizeBytes          Constant cache size in bytes.
         * @param constantL1FetchGranularityBytes fetch granularity.
         * @param l1MissPenalty                Penalty in cycles for an L1 miss.
         * @return Number of cores per cache.
         */
        uint32_t measureConstantL1Amount(size_t constantL1SizeBytes, size_t constantL1FetchGranularityBytes, double l1MissPenalty);
    }
}