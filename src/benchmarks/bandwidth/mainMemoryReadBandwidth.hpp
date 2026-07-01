#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure peak main memory read bandwidth.
     *
     * @param mainMemorySizeBytes Working set size in bytes.
     * @return Bandwidth in GiB/s.
     */
    double measureMainMemoryReadBandwidth(size_t mainMemorySizeBytes);

    /**
     * @brief Measure main memory read bandwidth with optimal number search for threads, blocks and reps.
     *
     * @param mainMemorySizeBytes Total device memory in bytes (the working set is derived from it).
     * @return Bandwidth in GiB/s and the optimal configuration (full sweep grid).
     */
    CacheBandwidthResult measureMainMemoryReadBandwidthSweep(size_t mainMemorySizeBytes);
}