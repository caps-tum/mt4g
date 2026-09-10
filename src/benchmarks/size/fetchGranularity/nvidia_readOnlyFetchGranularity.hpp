#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
         /**
          * @brief Determine fetch granularity of the read-only cache.
          *
          * @return Detected granularity in bytes.
          */
         CacheSizeResult measureReadOnlyFetchGranularity();
    }
}