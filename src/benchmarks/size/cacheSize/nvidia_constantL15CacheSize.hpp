#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure size of the constant L1.5 cache.
         *
         * @param cacheFetchGranularityBytes Line or sector size to use.
         * @return Detected cache size in bytes.
         */
        CacheSizeResult measureConstantL15Size(size_t cacheFetchGranularityBytes);
    }
}