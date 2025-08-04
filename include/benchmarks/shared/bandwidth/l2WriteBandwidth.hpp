#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure achievable L2 write bandwidth.
     *
     * @param l2SizeBytes Size of the L2 cache in bytes used for the test.
     * @return Bandwidth in GiB/s.
     */
    double measureL2WriteBandwidth(size_t l2SizeBytes);
}