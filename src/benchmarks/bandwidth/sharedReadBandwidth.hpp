#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable shared memory read bandwidth of single CU / SM.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s.
     */
    double measureSharedReadBandwidth(uint32_t arraySizeBytes);

    /**
     * @brief Measure achievable shared memory read bandwidth of single CU with optimal number search for threads, blocks and reps.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s and the optimal configuration.
     */
    CacheBandwidthResult measureSharedReadBandwidthSweep(uint32_t arraySizeBytes);
}