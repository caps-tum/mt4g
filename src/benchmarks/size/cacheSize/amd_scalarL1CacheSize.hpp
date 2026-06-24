#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the effective size of the scalar L1 cache.
         *
         * @param cacheFetchGranularityBytes Line or sector size to use.
         * @return Detected cache size in bytes.
         */
        CacheSizeResult measureScalarL1Size(size_t cacheFetchGranularityBytes);
    }
}