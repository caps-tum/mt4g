#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable L1 write bandwidth of a single CU.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s.
     */
    double measureL1WriteBandwidth(size_t arraySizeBytes);

    /**
     * @brief Measure achievable L1 write bandwidth of a single CU with optimal number search for threads, blocks and reps.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s and the optimal configuration.
     */
    CacheBandwidthResult measureL1WriteBandwidthSweep(size_t arraySizeBytes);
}
