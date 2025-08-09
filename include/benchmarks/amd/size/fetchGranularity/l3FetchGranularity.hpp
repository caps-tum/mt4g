#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Determine the fetch granularity of the L3 cache.
         *
         * @return Detected size in bytes.
         */
        CacheSizeResult measureL3FetchGranularity();
    }
}