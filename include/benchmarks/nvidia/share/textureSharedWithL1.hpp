#pragma once

#include <cstddef>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Check if texture and regular L1 caches share the same physical bank.
         *
         * @param textureCacheSizeBytes      Size of texture cache.
         * @param textureFetchGranularityBytes Line/sector size of texture cache.
         * @param textureLatency             Latency of texture cache hits.
         * @param textureMissPenalty         Miss penalty of texture cache.
         * @param l1CacheSizeBytes           Size of regular L1 cache.
         * @param l1FetchGranularityBytes    Line/sector size of regular L1.
         * @param l1Latency                  Latency of regular L1 hits.
         * @param l1MissPenalty              Miss penalty of regular L1.
         * @return True if both caches share the same bank, otherwise false.
         */
        bool measureTextureAndL1Shared(size_t textureCacheSizeBytes, size_t textureFetchGranularityBytes, double textureLatency, double textureMissPenalty, size_t l1CacheSizeBytes, size_t l1FetchGranularityBytes, double l1Latency, double l1MissPenalty);
    }
}