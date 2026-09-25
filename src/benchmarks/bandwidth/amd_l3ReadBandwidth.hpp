#pragma once

#include <cstddef>

#include "utils/hip/memory.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure achievable L3 read bandwidth on AMD GPUs.
         *
         * @param l2SizeBytes Size of the L2 cache in bytes used for the test.
         * @param l3SizeBytes Size of the L3 cache in bytes used for the test.
         * @return Bandwidth in GiB/s.
         */
        double measureL3ReadBandwidth(size_t l2SizeBytes, size_t l3SizeBytes);

        /**
         * @brief Measure achievable L3 read bandwidth on AMD GPUs with optimal configuration search.
         *
         * @param l2SizeBytes Size of the L2 cache in bytes used for the test.
         * @param l3SizeBytes Size of the L3 cache in bytes used for the test.
         * @param allocType Allocator used for the benchmarked source buffer.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureL3ReadBandwidthSweep(
            size_t l2SizeBytes,
            size_t l3SizeBytes,
            util::AllocatorType allocType = util::AllocatorType::HipMalloc);
    }
}
