#pragma once

#include <cstddef>

#include "typedef/cacheBandwidthResult.hpp"

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Measure constant L1 cache read bandwidth of a single MultiProcessor
         *        with optimal number search for threads and reps.
         *
         * @param arraySizeBytes Size of the constant array region in bytes used for the test.
         *                       Should fit into the constant L1 so the timed loop hits it.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureConstantL1ReadBandwidthSweep(size_t arraySizeBytes);
    }
}
