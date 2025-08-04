#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Determine how many cores share the texture cache.
         *
         * @param textureSizeBytes          Cache size in bytes.
         * @param textureFetchGranularityBytes fetch granularity.
         * @param textureMissPenalty        Penalty in cycles for a miss.
         * @return Number of cores per cache.
         */
        uint32_t measureTextureAmount(size_t textureSizeBytes, size_t textureFetchGranularityBytes, double textureMissPenalty);
    }
}