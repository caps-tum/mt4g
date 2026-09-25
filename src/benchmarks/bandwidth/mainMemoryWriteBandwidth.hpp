#pragma once

#include <cstddef>

#include "utils/hip/memory.hpp"

namespace benchmark {
    /**
     * @brief Measure peak main memory write bandwidth.
     *
     * @param mainMemorySizeBytes Working set size in bytes.
     * @return Bandwidth in GiB/s.
     */
    double measureMainMemoryWriteBandwidth(size_t mainMemorySizeBytes);

    /**
     * @brief Measure main memory write bandwidth with optimal number search for threads, blocks and reps.
     *
     * @param mainMemorySizeBytes Total device memory in bytes (the working set is derived from it).
     * @param allocType Allocator used for the benchmarked destination buffer.
     * @return Bandwidth in GiB/s and the optimal configuration (full sweep grid).
     */
    CacheBandwidthResult measureMainMemoryWriteBandwidthSweep(
        size_t mainMemorySizeBytes,
        util::AllocatorType allocType = util::AllocatorType::HipMalloc);
}
