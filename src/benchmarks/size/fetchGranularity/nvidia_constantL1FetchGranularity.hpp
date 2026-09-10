#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Determine fetch granularity of the constant L1 cache.
         *
         * @return Size in bytes.
         */
        CacheSizeResult measureConstantL1FetchGranularity();
    }
}