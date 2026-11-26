#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable L1 read bandwidth of a single CU / SM (one block), sweeping threads, reps and loads per batch.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s and the optimal configuration.
     */
    CacheBandwidthResult measureL1ReadBandwidthSweep(size_t arraySizeBytes);
}
