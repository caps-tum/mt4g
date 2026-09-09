#pragma once

#include <cstddef>

#include "typedef/cacheBandwidthResult.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure achievable constant L1.5 cache read bandwidth of a single
         *        MultiProcessor.
         *
         * @param arraySizeBytes Size of the constant array region in bytes used for the test.
         *                       Must exceed the constant L1 so the timed loop is served by
         *                       the L1.5 instead.
         * @param constantFetchGranularityBytes Stride between loads. Skips a full constant
         *                       cache line per access so no load can hit a line the constant
         *                       L1 already holds; clamped up to a line if smaller.
         * @return Bandwidth in GiB/s.
         */
        double measureConstantL15ReadBandwidth(size_t arraySizeBytes, size_t constantFetchGranularityBytes);

        /**
         * @brief Measure constant L1.5 cache read bandwidth of a single MultiProcessor
         *        with optimal number search for threads and reps.
         *
         * @param arraySizeBytes Size of the constant array region in bytes used for the test.
         *                       Must exceed the constant L1 so the timed loop is served by
         *                       the L1.5 instead.
         * @param constantFetchGranularityBytes Stride between loads. Skips a full constant
         *                       cache line per access so no load can hit a line the constant
         *                       L1 already holds; clamped up to a line if smaller.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureConstantL15ReadBandwidthSweep(size_t arraySizeBytes, size_t constantFetchGranularityBytes);
    }
}
