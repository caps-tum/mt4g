#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Determine fetch granularity of the texture cache.
         *
         * @return Detected granularity in bytes.
         */
        CacheSizeResult measureTextureFetchGranularity();
    }
}