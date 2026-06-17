#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure size of the constant memory L1 cache.
         *
         * @param cacheFetchGranularityBytes Line or sector size to use.
         * @return Detected cache size in bytes.
         */
        CacheSizeResult measureConstantL1Size(size_t cacheFetchGranularityBytes);
    }
}