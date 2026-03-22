#pragma once

#include <cstddef>

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure achievable sL1 read bandwidth on AMD GPUs.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s.
         */
        double measureScalarL1ReadBandwidth(size_t arraySizeBytes);

        /**
         * @brief Measure achievable sL1 read bandwidth on AMD GPUs with optimal configuration search.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureScalarL1ReadBandwidthSweep(size_t arraySizeBytes);
    }
}
