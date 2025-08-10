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
}