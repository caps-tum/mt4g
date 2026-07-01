#pragma once

#include <cstddef>

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure achievable L1 read bandwidth on AMD GPUs with optimal configuration search including block sweep.
         *
         * @param arraySizeBytes Size of the array in bytes used for the test.
         * @return Bandwidth in GiB/s and the optimal configuration.
         */
        CacheBandwidthResult measureL1ReadBandwidthBlockSweep(size_t arraySizeBytes);
    }
}
