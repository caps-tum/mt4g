#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Determine the fetch granularity of the scalar L1 cache.
         *
         * @return Detected size in bytes.
         */
        CacheSizeResult measureScalarL1FetchGranularity();
    }
}