#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure size of the read-only data cache.
         *
         * @param cacheFetchGranularityBytes Line or sector size used for measurement.
         * @return Detected cache size in bytes.
         */
        CacheSizeResult measureReadOnlySize(size_t cacheFetchGranularityBytes);
    }
}