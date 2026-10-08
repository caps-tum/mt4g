#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable L1 write bandwidth of a single CU (one block), sweeping threads and reps.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s and the optimal configuration.
     */
    CacheBandwidthResult measureL1WriteBandwidthSweep(size_t arraySizeBytes);

    namespace amd {
        /**
         * @brief Measure achievable L1 write bandwidth on AMD GPUs with a single block pinned to one CU, sweeping threads and reps.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureL1WriteBandwidthSweep(size_t arraySizeBytes);
    }
}
