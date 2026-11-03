#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable shared memory read bandwidth of single CU with static allocated shared memory and with optimal number search for threads, blocks and reps.
     *
     * @return Bandwidth in GiB/s and the optimal configuration.
     */
    CacheBandwidthResult measureSharedReadBandwidthStaticSweep();
}