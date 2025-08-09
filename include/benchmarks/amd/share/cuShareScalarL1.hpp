#pragma once

#include <cstddef>
#include <set>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure the number of compute units sharing a scalar L1 cache.
         *
         * @param l1SizeBytes          Cache size in bytes.
         * @param l1FetchGranularityBytes Line or sector size in bytes.
         * @return Number of compute units per cache.
         */
        std::set<std::set<uint32_t>>  measureCuShareScalarL1(size_t l1SizeBytes, size_t l1FetchGranularityBytes);
    }
}