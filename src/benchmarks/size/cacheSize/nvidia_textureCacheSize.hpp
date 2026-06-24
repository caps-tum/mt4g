#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure size of the texture cache.
         *
         * @param cacheFetchGranularityBytes Line or sector size used for measurement.
         * @return Detected cache size in bytes.
         */
        CacheSizeResult measureTextureSize(size_t cacheFetchGranularityBytes);
    }
}