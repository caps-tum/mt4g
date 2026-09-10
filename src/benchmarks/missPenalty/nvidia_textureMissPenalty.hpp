#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure the texture cache miss penalty.
         *
         * @param textureCacheSizeBytes      Size of the texture cache.
         * @param textureCacheLineSizeBytes  Line/sector size of the cache.
         * @param textureLatency             Latency of texture cache hits.
         * @return Miss penalty in cycles.
         */
        double measureTextureMissPenalty(size_t textureCacheSizeBytes, size_t textureCacheLineSizeBytes, double textureLatency);
   }
}