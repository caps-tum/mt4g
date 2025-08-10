#pragma once

#include <cstddef>

namespace benchmark {
    /**
     * @brief Measure peak main memory write bandwidth.
     *
     * @param mainMemorySizeBytes Working set size in bytes.
     * @return Bandwidth in GiB/s.
     */
    double measureMainMemoryWriteBandwidth(size_t mainMemorySizeBytes);
}