#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable L1 read bandwidth of single CU.
     *
     * @param arraySizeBytes Size of the array in bytes used for the test.
     * @return Bandwidth in GiB/s.
     */
    double measureL1ReadBandwidthCU(size_t arraySizeBytes);
}
