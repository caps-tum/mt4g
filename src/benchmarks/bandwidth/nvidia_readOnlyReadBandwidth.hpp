#pragma once

#include <cstddef>

#include "typedef/cacheBandwidthResult.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure achievable read-only cache read bandwidth of a single
         *        MultiProcessor.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s.
         */
        double measureReadOnlyReadBandwidth(size_t arraySizeBytes);

        /**
         * @brief Measure read-only cache read bandwidth of a single MultiProcessor
         *        with optimal number search for threads and reps.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureReadOnlyReadBandwidthSweep(size_t arraySizeBytes);
    }
}
