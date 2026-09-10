#pragma once
#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure the penalty incurred on an L2 cache miss.
     *
     * The cache is first evicted using L1-bypassing loads. The measured
     * access latency is then compared against the provided L2 hit latency.
     *
     * @param l2CacheSizeBytes     Total size of the L2 cache in bytes.
     * @param l2CacheLineSizeBytes Cache line or sector size in bytes.
     * @param l2Latency            Previously measured L2 hit latency.
     * @return Additional latency caused by an L2 miss.
     */
    double measureL2MissPenalty(size_t l2CacheSizeBytes,
                                size_t l2CacheLineSizeBytes,
                                double l2Latency);
}
