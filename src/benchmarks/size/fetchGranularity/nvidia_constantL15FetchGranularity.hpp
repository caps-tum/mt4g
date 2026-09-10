#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Determine fetch granularity of the constant L1.5 cache.
         *
         * @param constantL1FetchGranularityBytes Current granularity guess.
         * @return Detected granularity in bytes.
         */
        CacheSizeResult measureConstantL15FetchGranularity(size_t constantL1FetchGranularityBytes);
    }
}