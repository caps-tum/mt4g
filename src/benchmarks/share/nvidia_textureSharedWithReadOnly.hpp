#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Check if texture and read-only caches share the same physical bank.
         *
         * @param textureCacheSizeBytes      Size of texture cache.
         * @param textureFetchGranularityBytes Line/sector size of texture cache.
         * @param textureLatency             Latency of texture hits.
         * @param textureMissPenalty         Miss penalty of texture cache.
         * @param readOnlyCacheSizeBytes     Size of read-only cache.
         * @param readOnlyFetchGranularityBytes Line/sector size of read-only cache.
         * @param readOnlyLatency            Latency of read-only hits.
         * @param readOnlyMissPenalty        Miss penalty of read-only cache.
         * @return True if both caches share the same bank, otherwise false.
         */
        bool measureTextureAndReadOnlyShared(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, double textureLatency, double textureMissPenalty, size_t readOnlyCacheSizeBytes, size_t readOnlyFetchGranularityBytes, double readOnlyLatency, double readOnlyMissPenalty);
    }
}