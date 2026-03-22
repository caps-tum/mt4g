#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable L1 read bandwidth of single CU.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s.
     */
    double measureL1ReadBandwidth(size_t arraySizeBytes);

    /**
     * @brief Measure achievable L1 read bandwidth of single CU with optimal number search for threads, blocks and reps.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s and the optimal configuration.
     */
    CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes);
}
