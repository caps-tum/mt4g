#pragma once

#include <cstddef>

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure achievable L3 read bandwidth on AMD GPUs.
         *
         * @param l3SizeBytes Size of the L3 cache in bytes used for the test.
         * @return Bandwidth in GiB/s.
         */
        double measureL3ReadBandwidth(size_t l3SizeBytes);
    }
}
